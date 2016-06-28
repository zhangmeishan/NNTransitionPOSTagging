/*
 * APBeamSearcher.h
 *
 *  Created on: Mar 25, 2015
 *      Author: mszhang
 */

#ifndef SRC_APBeamSearcher_H_
#define SRC_APBeamSearcher_H_

#include <hash_set>
#include <iostream>

#include <assert.h>
#include "Feature.h"
#include "FeatureExtraction.h"
#include "N3L.h"
#include "State.h"
#include "Action.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//re-implementation of Yue and Clark ACL (2007)
class APBeamSearcher {

public:
	APBeamSearcher() {
		_dropOut = 0.5;
	}
	~APBeamSearcher() {
	}

public:
	AvgPerceptron1O<cpu> _splayer_output;

	FeatureExtraction fe;

	int _linearfeatSize;

	Metric _eval;

	dtype _dropOut;

	enum {
		BEAM_SIZE = 16, MAX_SENTENCE_SIZE = 512
	};

public:

	inline void addToFeatureAlphabet(hash_map<string, int> feat_stat, int featCutOff = 0) {
		fe.addToFeatureAlphabet(feat_stat, featCutOff);
	}

	inline void addToWordAlphabet(hash_map<string, int> word_stat, int wordCutOff = 0) {
		fe.addToWordAlphabet(word_stat, wordCutOff);
	}

	inline void addToCharAlphabet(hash_map<string, int> char_stat, int charCutOff = 0) {
		fe.addToCharAlphabet(char_stat, charCutOff);
	}

	inline void addToBiCharAlphabet(hash_map<string, int> bichar_stat, int bicharCutOff = 0) {
		fe.addToBiCharAlphabet(bichar_stat, bicharCutOff);
	}

	inline void addToActionAlphabet(hash_map<string, int> action_stat) {
		fe.addToActionAlphabet(action_stat);
	}

	inline void addToPostagAlphabet(hash_map<string, int> pos_stat) {
		fe.addToPostagAlphabet(pos_stat);
	}

	inline void setAlphaIncreasing(bool bAlphaIncreasing) {
		fe.setAlphaIncreasing(bAlphaIncreasing);
	}

	inline void initAlphabet() {
		fe.initAlphabet();
	}

	inline void loadAlphabet() {
		fe.loadAlphabet();
	}

	inline void extractFeature(const CStateItem * curState, const CAction& nextAC, Feature& feat) {
		fe.extractFeature(curState, nextAC, feat);
	}

public:

	inline void init() {
		_linearfeatSize = fe._featAlphabet.size();

		_splayer_output.initial(_linearfeatSize, 10);
	}

	inline void release() {
		_splayer_output.release();
	}

	dtype train(const std::vector<Instance>& sentences, const vector<vector<CAction> >& goldACs) {
		fe.setFeatureFormat(false);
		fe.setFeatAlphaIncreasing(true);
		_eval.reset();
		dtype cost = 0.0;
		for (int idx = 0; idx < sentences.size(); idx++) {
			cost += trainOneExample(sentences[idx], goldACs[idx], sentences.size());
		}

		return cost;
	}

	// scores do not accumulate together...., big bug, refine it tomorrow or at thursday.
	dtype trainOneExample(const Instance& sentence, const vector<CAction>& goldAC, int num) {
		if (sentence.charsize() >= MAX_SENTENCE_SIZE)
			return 0.0;
		static CStateItem lattice[(MAX_SENTENCE_SIZE + 1) * (BEAM_SIZE + 1)];
		static CStateItem * lattice_index[MAX_SENTENCE_SIZE + 1];

		int length = sentence.charsize();
		dtype cost = 0.0, curcost = 0.0;
		;
		dtype score = 0.0;

		const static CStateItem *pGenerator;
		const static CStateItem *pBestGen;
		static CStateItem *correctState;

		static bool bCorrect;  // used in learning for early update
		static int index, tmp_i, tmp_j, tmp_k;
		static CAction correct_action;
		static bool correct_action_scored, correct_action_contained;
		static std::vector<CAction> actions; // actions to apply for a candidate
		static NRHeap<CScoredStateAction, CScoredStateAction_Compare> beam(BEAM_SIZE);
		static CScoredStateAction scored_action; // used rank actions
		static CScoredStateAction scored_correct_action;
		static vector<dtype> normalizedScores;

		lattice_index[0] = lattice;
		lattice_index[1] = lattice + 1;
		lattice_index[0]->clear();
		lattice_index[0]->initSentence(&sentence.chars, &sentence.candidateLabels);

		index = 0;

		correctState = lattice_index[0];

		while (true) {
			++index;
			lattice_index[index + 1] = lattice_index[index];
			beam.clear();
			pBestGen = 0;
			correct_action = goldAC[index - 1];
			bCorrect = false;
			correct_action_scored = false;

			//std::cout << "check beam start" << std::endl;
			for (pGenerator = lattice_index[index - 1]; pGenerator != lattice_index[index]; ++pGenerator) {
				pGenerator->getCandidateActions(actions, fe._postagAlphabet);
				if (pGenerator == correctState) {
					correct_action_contained = false;
					for (tmp_j = 0; tmp_j < actions.size(); ++tmp_j) {
						if (actions[tmp_j] == correct_action) {
							correct_action_contained = true;
						}
					}
					if (!correct_action_contained) {
						actions.push_back(correct_action);
					}
				}
				for (tmp_j = 0; tmp_j < actions.size(); ++tmp_j) {
					scored_action.action = actions[tmp_j];
					scored_action.item = pGenerator;
					fe.extractFeature(pGenerator, actions[tmp_j], scored_action.feat, 0, 0, _dropOut);
					_splayer_output.ComputeForwardScore(scored_action.feat._nSparseFeat, scored_action.score, true);
					scored_action.score += pGenerator->_score;
					beam.add_elem(scored_action);

					if (pGenerator == correctState && actions[tmp_j] == correct_action) {
						scored_correct_action = scored_action;
						correct_action_scored = true;
						//std::cout << "add gold finish" << std::endl;
					} else {
						//std::cout << "add finish" << std::endl;
					}

				}
			}

			//std::cout << "check beam start" << std::endl;
			for (tmp_j = 0; tmp_j < beam.elemsize(); ++tmp_j) { // insert from
				pGenerator = beam[tmp_j].item;
				pGenerator->move(lattice_index[index + 1], beam[tmp_j].action, fe._postagAlphabet);
				lattice_index[index + 1]->_score = beam[tmp_j].score;
				lattice_index[index + 1]->_curFeat.copy(beam[tmp_j].feat);

				if (pBestGen == 0 || lattice_index[index + 1]->_score > pBestGen->_score) {
					pBestGen = lattice_index[index + 1];
				}
				if (pGenerator == correctState && beam[tmp_j].action == correct_action) {
					correctState = lattice_index[index + 1];
					bCorrect = true;
				}

				++lattice_index[index + 1];
			}
			//std::cout << "check beam finish" << std::endl;

			if (pBestGen->IsTerminated())
				break; // while

			// update items if correct item jump out of the agenda

			if (!bCorrect) {
				// note that if bCorrect == true then the correct state has
				// already been updated, and the new value is one of the new states
				// among the newly produced from lattice[index+1].
				correctState->move(lattice_index[index + 1], correct_action, fe._postagAlphabet);
				correctState = lattice_index[index + 1];
				lattice_index[index + 1]->_score = scored_correct_action.score;
				lattice_index[index + 1]->_curFeat.copy(scored_correct_action.feat);

				assert(correct_action_scored); // scored_correct_act valid

				normalize(beam, correctState, normalizedScores, false);
				tmp_j = 0;
				for (pGenerator = lattice_index[index]; pGenerator != lattice_index[index + 1]; ++pGenerator) {
					if (correctState->_score >= pGenerator->_score - 1e-10) {
						tmp_j++;
						continue;
					}
					curcost = backPropagationStates(pBestGen, correctState, normalizedScores[tmp_j] / num, -normalizedScores[tmp_j] / num);
					if (curcost < 0) {
						std::cout << "strange ..." << std::endl;
					}
					cost = cost + curcost;
					tmp_j++;
				}
				if (tmp_j != normalizedScores.size()) {
					std::cout << "beam number error" << std::endl;
				}
				_eval.correct_label_count += index;
				_eval.overall_label_count += length + 1;
				return cost;
			}

		}

		// make sure that the correct item is stack top finally
		if (pBestGen != correctState) {
			if (!bCorrect) {
				correctState->move(lattice_index[index + 1], correct_action, fe._postagAlphabet);
				correctState = lattice_index[index + 1];
				lattice_index[index + 1]->_score = scored_correct_action.score;
				lattice_index[index + 1]->_curFeat.copy(scored_correct_action.feat);
				assert(correct_action_scored); // scored_correct_act valid
			}
			normalize(beam, correctState, normalizedScores, false);
			tmp_j = 0;
			for (pGenerator = lattice_index[index]; pGenerator != lattice_index[index + 1]; ++pGenerator) {
				if (correctState->_score >= pGenerator->_score - 1e-10) {
					tmp_j++;
					continue;
				}
				curcost = backPropagationStates(pBestGen, correctState, normalizedScores[tmp_j] / num, -normalizedScores[tmp_j] / num);
				if (curcost < 0) {
					std::cout << "strange ..." << std::endl;
				}
				cost = cost + curcost;
				tmp_j++;
			}
			if (tmp_j != normalizedScores.size()) {
				std::cout << "beam number error" << std::endl;
			}
			_eval.correct_label_count += length;
			_eval.overall_label_count += length + 1;
		} else {
			_eval.correct_label_count += length + 1;
			_eval.overall_label_count += length + 1;
		}

		return cost;
	}

	bool decode(const Instance& sentence, CResult& result) {
		setAlphaIncreasing(false);
		if (sentence.charsize() >= MAX_SENTENCE_SIZE)
			return false;
		static CStateItem lattice[(MAX_SENTENCE_SIZE + 1) * (BEAM_SIZE + 1)];
		static CStateItem *lattice_index[MAX_SENTENCE_SIZE + 1];
		srand(1);

		int length = sentence.charsize();
		dtype score = 0.0;

		const static CStateItem *pGenerator;
		const static CStateItem *pBestGen;

		static int index, tmp_i, tmp_j, tmp_k;
		static std::vector<CAction> actions; // actions to apply for a candidate
		static NRHeap<CScoredStateAction, CScoredStateAction_Compare> beam(BEAM_SIZE);
		static CScoredStateAction scored_action; // used rank actions
		static Feature feat;

		lattice_index[0] = lattice;
		lattice_index[1] = lattice + 1;
		lattice_index[0]->clear();
		lattice_index[0]->initSentence(&sentence.chars, &sentence.candidateLabels);

		index = 0;

		while (true) {
			++index;
			lattice_index[index + 1] = lattice_index[index];
			beam.clear();
			pBestGen = 0;

			//std::cout << index << std::endl;
			for (pGenerator = lattice_index[index - 1]; pGenerator != lattice_index[index]; ++pGenerator) {
				pGenerator->getCandidateActions(actions, fe._postagAlphabet);
				for (tmp_j = 0; tmp_j < actions.size(); ++tmp_j) {
					scored_action.action = actions[tmp_j];
					scored_action.item = pGenerator;
					fe.extractFeature(pGenerator, actions[tmp_j], scored_action.feat);
					_splayer_output.ComputeForwardScore(scored_action.feat._nSparseFeat, scored_action.score);
					scored_action.score += pGenerator->_score;
					beam.add_elem(scored_action);
				}
			}

			if (beam.elemsize() == 0) {
				std::cout << "error: beam size zero!" << std::endl;
				for (int idx = 0; idx < sentence.charsize(); idx++) {
					std::cout << sentence.chars[idx] << std::endl;
				}
				std::cout << "" << std::endl;
				return false;
			}

			for (tmp_j = 0; tmp_j < beam.elemsize(); ++tmp_j) { // insert from
				pGenerator = beam[tmp_j].item;
				pGenerator->move(lattice_index[index + 1], beam[tmp_j].action, fe._postagAlphabet);
				lattice_index[index + 1]->_score = beam[tmp_j].score;

				if (pBestGen == 0 || lattice_index[index + 1]->_score > pBestGen->_score) {
					pBestGen = lattice_index[index + 1];
				}

				++lattice_index[index + 1];
			}

			if (pBestGen->IsTerminated())
				break; // while

		}

		pBestGen->getSegPosResults(result);

		return true;
	}
	// normalize scores
	void normalize(const NRHeap<CScoredStateAction, CScoredStateAction_Compare> &beam, const CStateItem *correctState, vector<dtype>& normalizedScores,
			bool bMax = true) {
		dtype maxScore = 0.0;
		int index = -1;
		static int tmp_j;
		int count = beam.elemsize();
		normalizedScores.resize(count);
		for (tmp_j = 0; tmp_j < count; ++tmp_j) {
			if (index == -1 || beam[tmp_j].score > maxScore) {
				index = tmp_j;
				maxScore = beam[tmp_j].score;
			}
			normalizedScores[tmp_j] = d_zero;
		}

		dtype sumScore = 0.0;
		int above_correct_count = 0;
		for (tmp_j = 0; tmp_j < count; ++tmp_j) {
			if (beam[tmp_j].score < correctState->_score - 1e-10)
				continue;
			normalizedScores[tmp_j] = exp(beam[tmp_j].score - maxScore);
			sumScore += normalizedScores[tmp_j];
			above_correct_count++;
		}

		//std::cout << "debug:";
		int select = index;
		//int select = rand()%count;
		for (tmp_j = 0; tmp_j < count; ++tmp_j) {
			//normalizedScores[tmp_j] = normalizedScores[tmp_j]/sumScore;
			//std::cout << " " << normalizedScores[tmp_j];
			if (bMax)
				normalizedScores[tmp_j] = (tmp_j == select) ? 1.0 : 0.0;
			else
				normalizedScores[tmp_j] = normalizedScores[tmp_j] / sumScore;
			//normalizedScores[tmp_j] = normalizedScores[tmp_j] < 1e-10 ? d_zero : 1.0 / above_correct_count;
		}
		//std::cout << std::endl;

	}

	dtype backPropagationStates(const CStateItem *pPredState, const CStateItem *pGoldState, dtype predLoss, dtype goldLoss) {
		if (pPredState == pGoldState)
			return 0.0;

		if (pPredState->_nextPosition != pGoldState->_nextPosition) {
			std::cout << "state align error" << std::endl;
		}
		dtype delta = 0.0;
		dtype predscore, goldscore;
		_splayer_output.ComputeForwardScore(pPredState->_curFeat._nSparseFeat, predscore, true);
		_splayer_output.ComputeForwardScore(pGoldState->_curFeat._nSparseFeat, goldscore, true);

		delta = predscore - goldscore;

		_splayer_output.ComputeBackwardLoss(pPredState->_curFeat._nSparseFeat, predLoss);
		_splayer_output.ComputeBackwardLoss(pGoldState->_curFeat._nSparseFeat, goldLoss);

		//currently we use a uniform loss
		delta += backPropagationStates(pPredState->_prevState, pGoldState->_prevState, predLoss, goldLoss);

		dtype compare_delta = pPredState->_score - pGoldState->_score;
		if (abs(delta - compare_delta) > 0.01) {
			std::cout << "delta=" << delta << "\t, compare_delta=" << compare_delta << std::endl;
		}

		return delta;
	}

	void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
		_splayer_output.updateAdaGrad(nnRegular, adaAlpha, adaEps);
	}

	void writeModel();

	void loadModel();

public:

	inline void resetEval() {
		_eval.reset();
	}

	inline void setDropValue(dtype dropOut) {
		_dropOut = dropOut;
	}

};

#endif /* SRC_APBeamSearcher_H_ */
