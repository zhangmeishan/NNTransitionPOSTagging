/*
 * LSTMBeamSearcher.h
 *
 *  Created on: Jan 25, 2016
 *      Author: mszhang
 */

#ifndef SRC_LSTMBeamSearcher_H_
#define SRC_LSTMBeamSearcher_H_

#include <hash_set>
#include <iostream>

#include <assert.h>
#include "Feature.h"
#include "FeatureExtraction.h"
#include "DenseFeature.h"
#include "DenseFeatureChar.h"
#include "N3L.h"
#include "State.h"
#include "Action.h"

#define LSTM_ALG LSTM_STD

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//re-implementation of Yue and Clark ACL (2007)

class LSTMBeamSearcher {
public:
	LSTMBeamSearcher() {
		_dropOut = 0.5;
		_delta = 0.2;
		_buffer = 6;
	}
	~LSTMBeamSearcher() {
	}

public:
	//SparseUniLayer1O<cpu> _splayer_output;

	LookupTable<cpu> _chars;
	LookupTable<cpu> _bichars;
	UniLayer<cpu> _nnlayer_char_hidden;
	LSTM_ALG<cpu> _char_left_rnn;
	LSTM_ALG<cpu> _char_right_rnn;

	LookupTable<cpu> _allwords;
	UniLayer<cpu> _nnlayer_word_hidden;
	LSTM_ALG<cpu> _word_rnn;

	LookupTable<cpu> _postags;
	UniLayer<cpu> _nnlayer_postag_hidden;
	LSTM_ALG<cpu> _postag_rnn;

	LookupTable<cpu> _actions;
	UniLayer<cpu> _nnlayer_action_hidden;
	LSTM_ALG<cpu> _action_rnn;

	UniLayer<cpu> _nnlayer_sep_hidden;
	vector<UniLayer1O<cpu> > _nnlayer_sep_output;

	UniLayer<cpu> _nnlayer_app_hidden;
	UniLayer1O<cpu> _nnlayer_app_output;

	int _allwordSize;
	int _allwordDim;
	int _wordNgram;
	int _wordRepresentDim;
	int _wordHiddenSize;
	int _wordRNNHiddenSize;

	int _charSize, _biCharSize;
	int _charDim, _biCharDim;
	int _charcontext, _charwindow;
	int _charRepresentDim;
	int _charHiddenSize;
	int _charRNNHiddenSize;

	int _postagSize;
	int _postagDim;
	int _postagNgram;
	int _postagRepresentDim;
	int _postagHiddenSize;
	int _postagRNNHiddenSize;

	int _actionSize;
	int _actionDim;
	int _actionNgram;
	int _actionRepresentDim;
	int _actionHiddenSize;
	int _actionRNNHiddenSize;

	int _sep_hiddenOutSize;
	int _sep_hiddenInSize;
	int _app_hiddenOutSize;
	int _app_hiddenInSize;

	FeatureExtraction fe;

	int _linearfeatSize;

	Metric _eval;

	dtype _dropOut;

	dtype _delta;

	int _buffer;

	enum {
		BEAM_SIZE = 16, MAX_SENTENCE_SIZE = 512
	};

public:

	inline void addToFeatureAlphabet(hash_map<string, int> feat_stat, int featCutOff = 0) {
		fe.addToFeatureAlphabet(feat_stat, featCutOff);
	}

	inline void addToAllWordAlphabet(hash_map<string, int> allword_stat, int allwordCutOff = 0) {
		fe.addToAllWordAlphabet(allword_stat, allwordCutOff);
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

	inline void extractFeature(const CStateItem* curState, const CAction& nextAC, Feature& feat) {
		fe.extractFeature(curState, nextAC, feat);
	}

public:

	inline void init(const NRMat<dtype>& allwordEmb, int wordNgram, int wordHiddenSize, int wordRNNHiddenSize,
			const NRMat<dtype>& charEmb, const NRMat<dtype>& bicharEmb, int charcontext, int charHiddenSize, int charRNNHiddenSize,
			const NRMat<dtype>& postagEmb, int postagNgram, int postagHiddenSize, int postagRNNHiddenSize,
			const NRMat<dtype>& actionEmb, int actionNgram, int actionHiddenSize, int actionRNNHiddenSize,
			int sep_hidden_out_size, int app_hidden_out_size, dtype delta) {
		_delta = delta;
		_linearfeatSize = 3 * fe._featAlphabet.size();

		_charSize = fe._charAlphabet.size();
		if (_charSize != charEmb.nrows())
			std::cout << "char number does not match for initialization of char emb table" << std::endl;
		_biCharSize = fe._bicharAlphabet.size();
		if (_biCharSize != bicharEmb.nrows())
			std::cout << "bichar number does not match for initialization of bichar emb table" << std::endl;
		_charDim = charEmb.ncols();
		_biCharDim = bicharEmb.ncols();
		_charcontext = charcontext;
		_charwindow = 2 * charcontext + 1;
		_charRepresentDim = (_charDim + _biCharDim) * _charwindow;

		_allwordSize = fe._allwordAlphabet.size();
		if (_allwordSize != allwordEmb.nrows())
			std::cout << "allword number does not match for initialization of allword emb table" << std::endl;
		_allwordDim = allwordEmb.ncols();
		_wordNgram = wordNgram;
		_wordRepresentDim = _wordNgram * _allwordDim;

		_postagSize = fe._postagAlphabet.size();
		if (_postagSize != postagEmb.nrows())
			std::cout << "postag number does not match for initialization of postag emb table" << std::endl;
		_postagDim = postagEmb.ncols();
		_postagNgram = postagNgram;
		_postagRepresentDim = _postagNgram * _postagDim;

		_actionSize = fe._actionAlphabet.size();
		if (_actionSize != actionEmb.nrows())
			std::cout << "action number does not match for initialization of action emb table" << std::endl;
		_actionDim = actionEmb.ncols();
		_actionNgram = actionNgram;
		_actionRepresentDim = _actionNgram * _actionDim;

		_wordHiddenSize = wordHiddenSize;
		_wordRNNHiddenSize = wordRNNHiddenSize;

		_charHiddenSize = charHiddenSize;
		_charRNNHiddenSize = charRNNHiddenSize;

		_actionHiddenSize = actionHiddenSize;
		_actionRNNHiddenSize = actionRNNHiddenSize;

		_postagHiddenSize = postagHiddenSize;
		_postagRNNHiddenSize = postagRNNHiddenSize;

		_sep_hiddenInSize = _wordRNNHiddenSize + _actionRNNHiddenSize + 2 * _charRNNHiddenSize + _postagRNNHiddenSize;
		_sep_hiddenOutSize = sep_hidden_out_size;

		_app_hiddenInSize = _actionRNNHiddenSize + 2 * _charRNNHiddenSize;
		_app_hiddenOutSize = app_hidden_out_size;

		srand(0);
		//_splayer_output.initial(_linearfeatSize, rand());

		_chars.initial(charEmb);
		_chars.setEmbFineTune(true);
		_bichars.initial(bicharEmb);
		_bichars.setEmbFineTune(true);
		_nnlayer_char_hidden.initial(_charHiddenSize, _charRepresentDim, true, rand());
		_char_left_rnn.initial(_charRNNHiddenSize, _charHiddenSize, true, rand());
		_char_right_rnn.initial(_charRNNHiddenSize, _charHiddenSize, false, rand());

		_allwords.initial(allwordEmb);
		_allwords.setEmbFineTune(false);
		_nnlayer_word_hidden.initial(_wordHiddenSize, _wordRepresentDim, true, rand());
		_word_rnn.initial(_wordRNNHiddenSize, _wordHiddenSize, true, rand());

		_postags.initial(postagEmb);
		_postags.setEmbFineTune(true);
		_nnlayer_postag_hidden.initial(_postagHiddenSize, _postagRepresentDim, true, rand());
		_postag_rnn.initial(_postagRNNHiddenSize, _postagHiddenSize, true, rand());

		_actions.initial(actionEmb);
		_actions.setEmbFineTune(true);
		_nnlayer_action_hidden.initial(_actionHiddenSize, _actionRepresentDim, true, rand());
		_action_rnn.initial(_actionRNNHiddenSize, _actionHiddenSize, true, rand());

		_nnlayer_sep_hidden.initial(_sep_hiddenOutSize, _sep_hiddenInSize, true, rand());

		// sep + pos num && finish
		for (int idx = 0; idx < _postagSize; idx++) {
			UniLayer1O<cpu> sep_outlayer;
			sep_outlayer.initial(_sep_hiddenOutSize, rand());
			_nnlayer_sep_output.push_back(sep_outlayer);
		}

		_nnlayer_app_hidden.initial(_app_hiddenOutSize, _app_hiddenInSize, true, rand());
		_nnlayer_app_output.initial(_app_hiddenOutSize, rand());

	}

	inline void release() {
		//_splayer_output.release();

		_chars.release();
		_bichars.release();
		_nnlayer_char_hidden.release();
		_char_left_rnn.release();
		_char_right_rnn.release();

		_allwords.release();
		_nnlayer_word_hidden.release();
		_word_rnn.release();

		_postags.release();
		_nnlayer_postag_hidden.release();
		_postag_rnn.release();

		_actions.release();
		_nnlayer_action_hidden.release();
		_action_rnn.release();

		_nnlayer_sep_hidden.release();
		for (int idx = 0; idx < _nnlayer_sep_output.size(); idx++) {
			_nnlayer_sep_output[idx].release();
		}
		_nnlayer_sep_output.clear();

		_nnlayer_app_hidden.release();
		_nnlayer_app_output.release();

	}

	dtype train(const std::vector<Instance>& sentences, const vector<vector<CAction> >& goldACs) {
		fe.setFeatureFormat(false);
		//setAlphaIncreasing(true);
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
		static CStateItem *lattice_index[MAX_SENTENCE_SIZE + 1];

		int length = sentence.charsize();
		dtype cost = 0.0;
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
		static DenseFeatureChar charFeat;
		static vector<dtype> normalizedScores;

		lattice_index[0] = lattice;
		lattice_index[1] = lattice + 1;
		lattice_index[0]->clear();
		lattice_index[0]->initSentence(&sentence.chars, &sentence.candidateLabels);

		index = 0;

		/*
		 Add Character bi rnn  here
		 */
		forwardStaticFeatures(sentence.chars, charFeat, true);

		correctState = lattice_index[0];

		while (true) {
			++index;
			lattice_index[index + 1] = lattice_index[index];
			beam.clear();
			pBestGen = 0;
			correct_action = goldAC[index - 1];
			bCorrect = false;
			correct_action_scored = false;

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
					//scored_action.clear(); // no need, because we will assign all members with new values for it
					scored_action.action = actions[tmp_j];
					scored_action.item = pGenerator;
					forwardDynamicFeatures(charFeat, scored_action, true);
					if (actions[tmp_j] != correct_action) {
						scored_action.score += _delta;
					}
					beam.add_elem(scored_action);

					if (pGenerator == correctState && actions[tmp_j] == correct_action) {
						scored_correct_action = scored_action;
						correct_action_scored = true;
					} else {
					}

				}
			}

			//std::cout << "check beam start" << std::endl;
			for (tmp_j = 0; tmp_j < beam.elemsize(); ++tmp_j) { // insert from
				pGenerator = beam[tmp_j].item;
				pGenerator->move(lattice_index[index + 1], beam[tmp_j].action, fe._postagAlphabet);
				lattice_index[index + 1]->_score = beam[tmp_j].score;
				lattice_index[index + 1]->_curFeat.copy(beam[tmp_j].feat);
				lattice_index[index + 1]->_nnfeat.copy(beam[tmp_j].nnfeat);

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
				lattice_index[index + 1]->_nnfeat.copy(scored_correct_action.nnfeat);

				assert(correct_action_scored); // scored_correct_act valid
				normalize(beam, correctState, normalizedScores, false);
				tmp_j = 0;
				for (pGenerator = lattice_index[index]; pGenerator != lattice_index[index + 1]; ++pGenerator) {
					if (pGenerator->_score < correctState->_score - 1e-10) {
						std::cout << "update error" << std::endl;
					}
					backwardContraryStates(pGenerator, correctState, charFeat, index, normalizedScores[tmp_j] / num);
					tmp_j++;
				}
				if (tmp_j != normalizedScores.size()) {
					std::cout << "beam number error" << std::endl;
				}
				backwardStaticFeatures(charFeat);
				//backwardContraryStates(pBestGen, correctState, charFeat, index, 1.0/num);

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
				lattice_index[index + 1]->_nnfeat.copy(scored_correct_action.nnfeat);
				assert(correct_action_scored); // scored_correct_act valid
			}

			normalize(beam, correctState, normalizedScores, false);
			tmp_j = 0;
			for (pGenerator = lattice_index[index]; pGenerator != lattice_index[index + 1]; ++pGenerator) {
				if (correctState->_score >= pGenerator->_score - 1e-10) {
					tmp_j++;
					continue;
				}
				backwardContraryStates(pGenerator, correctState, charFeat, index, normalizedScores[tmp_j] / num);
				tmp_j++;
			}
			if (tmp_j != normalizedScores.size()) {
				std::cout << "beam number error" << std::endl;
			}
			backwardStaticFeatures(charFeat);
			//backwardContraryStates(pBestGen, correctState, charFeat, index, 1.0/num);

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

		int length = sentence.charsize();
		dtype cost = 0.0;
		dtype score = 0.0;

		const static CStateItem *pGenerator;
		const static CStateItem *pBestGen;

		static int index, tmp_i, tmp_j, tmp_k;
		static std::vector<CAction> actions; // actions to apply for a candidate
		static NRHeap<CScoredStateAction, CScoredStateAction_Compare> beam(BEAM_SIZE);
		static CScoredStateAction scored_action; // used rank actions
		static Feature feat;
		static DenseFeatureChar charFeat;

		lattice_index[0] = lattice;
		lattice_index[1] = lattice + 1;
		lattice_index[0]->clear();
		lattice_index[0]->initSentence(&sentence.chars, &sentence.candidateLabels);

		index = 0;

		/*
		 Add Character bi rnn  here
		 */
		forwardStaticFeatures(sentence.chars, charFeat);

		while (true) {
			++index;
			lattice_index[index + 1] = lattice_index[index];
			beam.clear();
			pBestGen = 0;

			//std::cout << "check beam start" << std::endl;
			for (pGenerator = lattice_index[index - 1]; pGenerator != lattice_index[index]; ++pGenerator) {
				//std::cout << "new" << std::endl;
				//std::cout << pGenerator->str() << std::endl;
				pGenerator->getCandidateActions(actions, fe._postagAlphabet);
				for (tmp_j = 0; tmp_j < actions.size(); ++tmp_j) {
					//scored_action.clear();
					scored_action.action = actions[tmp_j];
					scored_action.item = pGenerator;

					forwardDynamicFeatures(charFeat, scored_action);

					beam.add_elem(scored_action);
				}

			}

			if (beam.elemsize() == 0) {
				std::cout << "error: beam size zero!" << std::endl;
				for (int idx = 0; idx < sentence.charsize(); idx++) {
					std::cout << sentence.chars[idx] << std::endl;
				}
				std::cout << "" << std::endl;
				charFeat.clear();
				return false;
			}

			for (tmp_j = 0; tmp_j < beam.elemsize(); ++tmp_j) { // insert from
				pGenerator = beam[tmp_j].item;
				pGenerator->move(lattice_index[index + 1], beam[tmp_j].action, fe._postagAlphabet);
				lattice_index[index + 1]->_score = beam[tmp_j].score;
				lattice_index[index + 1]->_nnfeat.copy(beam[tmp_j].nnfeat);

				if (pBestGen == 0 || lattice_index[index + 1]->_score > pBestGen->_score) {
					pBestGen = lattice_index[index + 1];
				}

				++lattice_index[index + 1];
			}

			if (pBestGen->IsTerminated())
				break; // while

		}

		pBestGen->getSegPosResults(result);

		charFeat.clear();

		return true;
	}

	void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
		//_splayer_output.updateAdaGrad(nnRegular, adaAlpha, adaEps);

		_chars.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_bichars.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_nnlayer_char_hidden.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_char_left_rnn.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_char_right_rnn.updateAdaGrad(nnRegular, adaAlpha, adaEps);

		_allwords.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_nnlayer_word_hidden.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_word_rnn.updateAdaGrad(nnRegular, adaAlpha, adaEps);

		_postags.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_nnlayer_postag_hidden.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_postag_rnn.updateAdaGrad(nnRegular, adaAlpha, adaEps);

		_actions.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_nnlayer_action_hidden.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_action_rnn.updateAdaGrad(nnRegular, adaAlpha, adaEps);

		_nnlayer_sep_hidden.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		for (int idx = 0; idx < _nnlayer_sep_output.size(); idx++) {
			_nnlayer_sep_output[idx].updateAdaGrad(nnRegular, adaAlpha, adaEps);
		}

		_nnlayer_app_hidden.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_nnlayer_app_output.updateAdaGrad(nnRegular, adaAlpha, adaEps);
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

private:
	void forwardDynamicFeatures(const DenseFeatureChar &charFeat, CScoredStateAction &scored_action, bool bTrain = false) {
		int length = charFeat._charnum;
		const CStateItem *pGenerator = scored_action.item;
		dtype score = 0.0;

		fe.extractFeature(pGenerator, scored_action.action, scored_action.feat, _wordNgram, _postagNgram, _actionNgram);
		scored_action.score = pGenerator->_score;
		//_splayer_output.ComputeForwardScore(scored_action.feat._nSparseFeat, score);
		//scored_action.score += score;

		scored_action.nnfeat.init(0, _allwordDim, 0, 0,
				_wordNgram, _wordHiddenSize, _wordRNNHiddenSize,
				_postagDim, _postagNgram, _postagHiddenSize, _postagRNNHiddenSize,
				_actionDim, _actionNgram, _actionHiddenSize, _actionRNNHiddenSize,
				_sep_hiddenInSize, _app_hiddenInSize, _sep_hiddenOutSize, _app_hiddenOutSize, _buffer, bTrain);

		//neural
		//action list
		for (int tmp_k = 0; tmp_k < _actionNgram; tmp_k++) {
			_actions.GetEmb(scored_action.feat._nActionFeat[tmp_k], scored_action.nnfeat._actionPrime[tmp_k]);
		}

		concat(scored_action.nnfeat._actionPrime, scored_action.nnfeat._actionRep);
		if (bTrain) {
			dropoutcol(scored_action.nnfeat._actionRepMask, _dropOut);
			scored_action.nnfeat._actionRep = scored_action.nnfeat._actionRep * scored_action.nnfeat._actionRepMask;
		}

		_nnlayer_action_hidden.ComputeForwardScore(scored_action.nnfeat._actionRep, scored_action.nnfeat._actionHidden);

		if (pGenerator->_nextPosition == 0) {
			_action_rnn.ComputeForwardScoreIncremental(scored_action.nnfeat._actionHidden,
					scored_action.nnfeat._actionRNNHiddenBuf[0], scored_action.nnfeat._actionRNNHiddenBuf[1], scored_action.nnfeat._actionRNNHiddenBuf[2],
					scored_action.nnfeat._actionRNNHiddenBuf[3], scored_action.nnfeat._actionRNNHiddenBuf[4], scored_action.nnfeat._actionRNNHiddenBuf[5],
					scored_action.nnfeat._actionRNNHidden);
		}
		else {
			_action_rnn.ComputeForwardScoreIncremental(pGenerator->_nnfeat._actionRNNHiddenBuf[4], pGenerator->_nnfeat._actionRNNHidden, scored_action.nnfeat._actionHidden,
					scored_action.nnfeat._actionRNNHiddenBuf[0], scored_action.nnfeat._actionRNNHiddenBuf[1], scored_action.nnfeat._actionRNNHiddenBuf[2],
					scored_action.nnfeat._actionRNNHiddenBuf[3], scored_action.nnfeat._actionRNNHiddenBuf[4], scored_action.nnfeat._actionRNNHiddenBuf[5],
					scored_action.nnfeat._actionRNNHidden);
		}

		//read word
		if (scored_action.action.isSeparate() || scored_action.action.isFinish()) {

			//word list
			for (int tmp_k = 0; tmp_k < _wordNgram; tmp_k++) {
				_allwords.GetEmb(scored_action.feat._nAllWordFeat[tmp_k], scored_action.nnfeat._allwordPrime[tmp_k]);
			}

			concat(scored_action.nnfeat._allwordPrime, scored_action.nnfeat._allwordRep);

			scored_action.nnfeat._wordUnitRep += scored_action.nnfeat._allwordRep;

			if (bTrain) {
				dropoutcol(scored_action.nnfeat._wordUnitRepMask, _dropOut);
				scored_action.nnfeat._wordUnitRep = scored_action.nnfeat._wordUnitRep * scored_action.nnfeat._wordUnitRepMask;
			}

			_nnlayer_word_hidden.ComputeForwardScore(scored_action.nnfeat._wordUnitRep, scored_action.nnfeat._wordHidden);

			const CStateItem * preSepState = pGenerator->_prevSepState;
			if (preSepState == 0) {
				_word_rnn.ComputeForwardScoreIncremental(scored_action.nnfeat._wordHidden,
						scored_action.nnfeat._wordRNNHiddenBuf[0], scored_action.nnfeat._wordRNNHiddenBuf[1], scored_action.nnfeat._wordRNNHiddenBuf[2],
						scored_action.nnfeat._wordRNNHiddenBuf[3], scored_action.nnfeat._wordRNNHiddenBuf[4], scored_action.nnfeat._wordRNNHiddenBuf[5],
						scored_action.nnfeat._wordRNNHidden);
			} else {
				_word_rnn.ComputeForwardScoreIncremental(preSepState->_nnfeat._wordRNNHiddenBuf[4], preSepState->_nnfeat._wordRNNHidden, scored_action.nnfeat._wordHidden,
						scored_action.nnfeat._wordRNNHiddenBuf[0], scored_action.nnfeat._wordRNNHiddenBuf[1], scored_action.nnfeat._wordRNNHiddenBuf[2],
						scored_action.nnfeat._wordRNNHiddenBuf[3], scored_action.nnfeat._wordRNNHiddenBuf[4], scored_action.nnfeat._wordRNNHiddenBuf[5],
						scored_action.nnfeat._wordRNNHidden);
			}

			//postag list
			for (int tmp_k = 0; tmp_k < _postagNgram; tmp_k++) {
				_postags.GetEmb(scored_action.feat._nPostagFeat[tmp_k], scored_action.nnfeat._postagPrime[tmp_k]);
			}

			concat(scored_action.nnfeat._postagPrime, scored_action.nnfeat._postagRep);

			if (bTrain) {
				dropoutcol(scored_action.nnfeat._postagRepMask, _dropOut);
				scored_action.nnfeat._postagRep = scored_action.nnfeat._postagRep * scored_action.nnfeat._postagRepMask;
			}

			_nnlayer_postag_hidden.ComputeForwardScore(scored_action.nnfeat._postagRep, scored_action.nnfeat._postagHidden);

			if (preSepState == 0) {
				_postag_rnn.ComputeForwardScoreIncremental(scored_action.nnfeat._postagHidden,
						scored_action.nnfeat._postagRNNHiddenBuf[0], scored_action.nnfeat._postagRNNHiddenBuf[1], scored_action.nnfeat._postagRNNHiddenBuf[2],
						scored_action.nnfeat._postagRNNHiddenBuf[3], scored_action.nnfeat._postagRNNHiddenBuf[4], scored_action.nnfeat._postagRNNHiddenBuf[5],
						scored_action.nnfeat._postagRNNHidden);
			} else {
				_postag_rnn.ComputeForwardScoreIncremental(preSepState->_nnfeat._postagRNNHiddenBuf[4], preSepState->_nnfeat._postagRNNHidden, scored_action.nnfeat._postagHidden,
						scored_action.nnfeat._postagRNNHiddenBuf[0], scored_action.nnfeat._postagRNNHiddenBuf[1], scored_action.nnfeat._postagRNNHiddenBuf[2],
						scored_action.nnfeat._postagRNNHiddenBuf[3], scored_action.nnfeat._postagRNNHiddenBuf[4], scored_action.nnfeat._postagRNNHiddenBuf[5],
						scored_action.nnfeat._postagRNNHidden);
			}

			//
			if (scored_action.item->_nextPosition < length) {
				concat(scored_action.nnfeat._wordRNNHidden, scored_action.nnfeat._actionRNNHidden, charFeat._charLeftRNNHidden[pGenerator->_nextPosition],
						charFeat._charRightRNNHidden[pGenerator->_nextPosition], scored_action.nnfeat._postagRNNHidden, scored_action.nnfeat._sepInHidden);
			} else {
				concat(scored_action.nnfeat._wordRNNHidden, scored_action.nnfeat._actionRNNHidden, charFeat._charRNNHiddenDummy,
						charFeat._charRNNHiddenDummy, scored_action.nnfeat._postagRNNHidden, scored_action.nnfeat._sepInHidden);
			}

			_nnlayer_sep_hidden.ComputeForwardScore(scored_action.nnfeat._sepInHidden, scored_action.nnfeat._sepOutHidden);
			if (scored_action.action.isFinish()) {
				_nnlayer_sep_output[0].ComputeForwardScore(scored_action.nnfeat._sepOutHidden, score);
			}
			else {
				_nnlayer_sep_output[scored_action.action._code - CAction::SEP].ComputeForwardScore(scored_action.nnfeat._sepOutHidden, score);
			}

		} else {
			concat(scored_action.nnfeat._actionRNNHidden, charFeat._charLeftRNNHidden[pGenerator->_nextPosition],
					charFeat._charRightRNNHidden[pGenerator->_nextPosition], scored_action.nnfeat._appInHidden);
			_nnlayer_app_hidden.ComputeForwardScore(scored_action.nnfeat._appInHidden, scored_action.nnfeat._appOutHidden);
			_nnlayer_app_output.ComputeForwardScore(scored_action.nnfeat._appOutHidden, score);
		}

		scored_action.score += score;
	}

	void forwardStaticFeatures(const std::vector<std::string>& chars, DenseFeatureChar &charFeat, bool bTrain = false) {
		int length = chars.size();
		charFeat.init(length, _charDim, _biCharDim, _charcontext, _charHiddenSize, _charRNNHiddenSize, _buffer, true);
		int unknownCharID = fe._charAlphabet[fe.unknownkey];
		for (int idx = 0; idx < length; idx++) {
			charFeat._charIds[idx] = fe._charAlphabet[chars[idx]];
			if (charFeat._charIds[idx] < 0)
				charFeat._charIds[idx] = unknownCharID;
		}

		int unknownBiCharID = fe._bicharAlphabet[fe.unknownkey];
		for (int idx = 0; idx < length; idx++) {
			charFeat._bicharIds[idx] = idx < length - 1 ? fe._bicharAlphabet[chars[idx] + chars[idx + 1]] : fe._bicharAlphabet[chars[idx] + fe.nullkey];
			if (charFeat._bicharIds[idx] < 0)
				charFeat._bicharIds[idx] = unknownBiCharID;
		}

		for (int idx = 0; idx < length; idx++) {
			_chars.GetEmb(charFeat._charIds[idx], charFeat._charprime[idx]);
			_bichars.GetEmb(charFeat._bicharIds[idx], charFeat._bicharprime[idx]);
			concat(charFeat._charprime[idx], charFeat._bicharprime[idx], charFeat._charpre[idx]);

			if (bTrain) {
				dropoutcol(charFeat._charpreMask[idx], _dropOut);
				charFeat._charpre[idx] = charFeat._charpre[idx] * charFeat._charpreMask[idx];
			}
		}

		windowlized(charFeat._charpre, charFeat._charInput, _charcontext);

		_nnlayer_char_hidden.ComputeForwardScore(charFeat._charInput, charFeat._charHidden);
		_char_left_rnn.ComputeForwardScore(charFeat._charHidden, charFeat._charLeftRNNHiddenBuf[0], charFeat._charLeftRNNHiddenBuf[1], charFeat._charLeftRNNHiddenBuf[2],
				charFeat._charLeftRNNHiddenBuf[3], charFeat._charLeftRNNHiddenBuf[4], charFeat._charLeftRNNHiddenBuf[5], charFeat._charLeftRNNHidden);
		_char_right_rnn.ComputeForwardScore(charFeat._charHidden, charFeat._charRightRNNHiddenBuf[0], charFeat._charRightRNNHiddenBuf[1], charFeat._charRightRNNHiddenBuf[2],
				charFeat._charRightRNNHiddenBuf[3], charFeat._charRightRNNHiddenBuf[4], charFeat._charRightRNNHiddenBuf[5], charFeat._charRightRNNHidden);
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
	// for training, back propagation
	void backwardContraryStates(const CStateItem *pBestGen, const CStateItem *correctState, DenseFeatureChar &charFeat, int steps, dtype absScore) {
		static DenseFeature pBestGenFeat, pGoldFeat;
		if (absScore < 1e-10)
			return;
		pBestGenFeat.init(pBestGen->_wordnum, steps, 0, _allwordDim, 0, 0,
				_wordNgram, _wordRepresentDim, _wordHiddenSize, _wordRNNHiddenSize,
				_postagDim, _postagNgram, _postagHiddenSize, _postagRNNHiddenSize,
				_actionDim, _actionNgram, _actionHiddenSize, _actionRNNHiddenSize, _buffer);
		pGoldFeat.init(correctState->_wordnum, steps, 0, _allwordDim, 0, 0,
				_wordNgram, _wordRepresentDim, _wordHiddenSize, _wordRNNHiddenSize,
				_postagDim, _postagNgram, _postagHiddenSize, _postagRNNHiddenSize,
				_actionDim, _actionNgram, _actionHiddenSize, _actionRNNHiddenSize, _buffer);

		backPropagationStates(pBestGen, correctState, absScore, -absScore, charFeat._charLeftRNNHidden_Loss, charFeat._charRightRNNHidden_Loss, charFeat._charRNNHiddenDummy_Loss, pBestGenFeat,
				pGoldFeat);

		pBestGenFeat.clear();
		pGoldFeat.clear();
	}

	void backwardStaticFeatures(DenseFeatureChar &charFeat) {
		_char_left_rnn.ComputeBackwardLoss(charFeat._charHidden, charFeat._charLeftRNNHiddenBuf[0], charFeat._charLeftRNNHiddenBuf[1], charFeat._charLeftRNNHiddenBuf[2],
				charFeat._charLeftRNNHiddenBuf[3], charFeat._charLeftRNNHiddenBuf[4], charFeat._charLeftRNNHiddenBuf[5], charFeat._charLeftRNNHidden,
				charFeat._charLeftRNNHidden_Loss, charFeat._charHidden_Loss);
		_char_right_rnn.ComputeBackwardLoss(charFeat._charHidden, charFeat._charRightRNNHiddenBuf[0], charFeat._charRightRNNHiddenBuf[1], charFeat._charRightRNNHiddenBuf[2],
				charFeat._charRightRNNHiddenBuf[3], charFeat._charRightRNNHiddenBuf[4], charFeat._charRightRNNHiddenBuf[5], charFeat._charRightRNNHidden,
				charFeat._charRightRNNHidden_Loss, charFeat._charHidden_Loss);
		_nnlayer_char_hidden.ComputeBackwardLoss(charFeat._charInput, charFeat._charHidden, charFeat._charHidden_Loss, charFeat._charInput_Loss);
		windowlized_backward(charFeat._charpre_Loss, charFeat._charInput_Loss, _charcontext);
		charFeat._charpre_Loss = charFeat._charpre_Loss * charFeat._charpreMask;

		for (int idx = 0; idx < charFeat._charnum; idx++) {
			unconcat(charFeat._charprime_Loss[idx], charFeat._bicharprime_Loss[idx], charFeat._charpre_Loss[idx]);
			_chars.EmbLoss(charFeat._charIds[idx], charFeat._charprime_Loss[idx]);
			_bichars.EmbLoss(charFeat._bicharIds[idx], charFeat._bicharprime_Loss[idx]);
		}

		//charFeat.zeroLoss();	
		charFeat.clear();
	}

	void backPropagationStates(const CStateItem *pPredState, const CStateItem *pGoldState, dtype predLoss, dtype goldLoss,
			Tensor<cpu, 3, dtype> charLeftRNNHidden_Loss, Tensor<cpu, 3, dtype> charRightRNNHidden_Loss, Tensor<cpu, 2, dtype> charRNNHiddenDummy_Loss,
			DenseFeature& predDenseFeature, DenseFeature& goldDenseFeature) {

		if (pPredState->_nextPosition != pGoldState->_nextPosition) {
			std::cout << "state align error" << std::endl;
		}

		static int position, word_position;

		//rnn loss computer together
		if (pPredState->_nextPosition == 0) {
			//predState
			if (pGoldState->_nextPosition != 0) {
				std::cout << "state align error !" << std::endl;
			}
			backwardDynamicRNNLoss(predDenseFeature);
			backwardDynamicRNNLoss(goldDenseFeature);

			return;
		}

		if (pPredState != pGoldState) {
			//predState
			computeStateLoss(pPredState, predLoss, predDenseFeature, charLeftRNNHidden_Loss, charRightRNNHidden_Loss, charRNNHiddenDummy_Loss);
			//goldState
			computeStateLoss(pGoldState, goldLoss, goldDenseFeature, charLeftRNNHidden_Loss, charRightRNNHidden_Loss, charRNNHiddenDummy_Loss);
		}

		//predState
		copyDynamicRNNValues(pPredState, predDenseFeature);
		//goldState
		copyDynamicRNNValues(pGoldState, goldDenseFeature);

		//currently we use a uniform loss
		backPropagationStates(pPredState->_prevState, pGoldState->_prevState, predLoss, goldLoss, charLeftRNNHidden_Loss,
				charRightRNNHidden_Loss, charRNNHiddenDummy_Loss, predDenseFeature, goldDenseFeature);

	}

	void backwardDynamicRNNLoss(DenseFeature& denseFeature) {
		//word rnn backpropagation
		_word_rnn.ComputeBackwardLoss(denseFeature._wordHidden,
				denseFeature._wordRNNHiddenBuf[0], denseFeature._wordRNNHiddenBuf[1], denseFeature._wordRNNHiddenBuf[2],
				denseFeature._wordRNNHiddenBuf[3], denseFeature._wordRNNHiddenBuf[4], denseFeature._wordRNNHiddenBuf[5],
				denseFeature._wordRNNHidden, denseFeature._wordRNNHiddenLoss, denseFeature._wordHiddenLoss);
		_nnlayer_word_hidden.ComputeBackwardLoss(denseFeature._wordUnitRep, denseFeature._wordHidden, denseFeature._wordHiddenLoss,
				denseFeature._wordUnitRepLoss);

		//postag rnn backpropagation
		_postag_rnn.ComputeBackwardLoss(denseFeature._postagHidden,
				denseFeature._postagRNNHiddenBuf[0], denseFeature._postagRNNHiddenBuf[1], denseFeature._postagRNNHiddenBuf[2],
				denseFeature._postagRNNHiddenBuf[3], denseFeature._postagRNNHiddenBuf[4], denseFeature._postagRNNHiddenBuf[5],
				denseFeature._postagRNNHidden, denseFeature._postagRNNHiddenLoss, denseFeature._postagHiddenLoss);
		_nnlayer_postag_hidden.ComputeBackwardLoss(denseFeature._postagRep, denseFeature._postagHidden, denseFeature._postagHiddenLoss,
				denseFeature._postagRepLoss);

		for (int idx = 0; idx < denseFeature._postagRepLoss.size(); idx++) {
			denseFeature._postagRepLoss[idx] = denseFeature._postagRepLoss[idx] * denseFeature._postagRepMask[idx];

			unconcat(denseFeature._postagPrimeLoss[idx], denseFeature._postagRepLoss[idx]);
			for (int tmp_k = 0; tmp_k < _postagNgram; tmp_k++) {
				_postags.EmbLoss(denseFeature._postagIds[idx][tmp_k], denseFeature._postagPrimeLoss[idx][tmp_k]);
			}
		}

		//action rnn backpropagation
		_action_rnn.ComputeBackwardLoss(denseFeature._actionHidden,
				denseFeature._actionRNNHiddenBuf[0], denseFeature._actionRNNHiddenBuf[1], denseFeature._actionRNNHiddenBuf[2],
				denseFeature._actionRNNHiddenBuf[3], denseFeature._actionRNNHiddenBuf[4], denseFeature._actionRNNHiddenBuf[5],
				denseFeature._actionRNNHidden, denseFeature._actionRNNHiddenLoss, denseFeature._actionHiddenLoss);
		_nnlayer_action_hidden.ComputeBackwardLoss(denseFeature._actionRep, denseFeature._actionHidden, denseFeature._actionHiddenLoss,
				denseFeature._actionRepLoss);

		for (int idx = 0; idx < denseFeature._actionRepLoss.size(); idx++) {
			denseFeature._actionRepLoss[idx] = denseFeature._actionRepLoss[idx] * denseFeature._actionRepMask[idx];

			unconcat(denseFeature._actionPrimeLoss[idx], denseFeature._actionRepLoss[idx]);
			for (int tmp_k = 0; tmp_k < _actionNgram; tmp_k++) {
				_actions.EmbLoss(denseFeature._actionIds[idx][tmp_k], denseFeature._actionPrimeLoss[idx][tmp_k]);
			}
		}
	}

	void computeStateLoss(const CStateItem *pState, dtype loss, DenseFeature& denseFeature,
			Tensor<cpu, 3, dtype> charLeftRNNHidden_Loss, Tensor<cpu, 3, dtype> charRightRNNHidden_Loss, Tensor<cpu, 2, dtype> charRNNHiddenDummy_Loss) {
		//sparse
		//_splayer_output.ComputeBackwardLoss(pState->_curFeat._nSparseFeat, loss);
		int length = charLeftRNNHidden_Loss.size(0);
		static int position, word_position;

		position = pState->_nextPosition - 1;

		if (pState->_lastAction.isSeparate() || pState->_lastAction.isFinish()) {
			if (pState->_lastAction.isFinish()) {
				_nnlayer_sep_output[0].ComputeBackwardLoss(pState->_nnfeat._sepOutHidden, loss, pState->_nnfeat._sepOutHiddenLoss, true);
			}
			else {
				_nnlayer_sep_output[pState->_lastAction._code - CAction::SEP].ComputeBackwardLoss(pState->_nnfeat._sepOutHidden, loss, pState->_nnfeat._sepOutHiddenLoss, true);
			}
			_nnlayer_sep_hidden.ComputeBackwardLoss(pState->_nnfeat._sepInHidden, pState->_nnfeat._sepOutHidden, pState->_nnfeat._sepOutHiddenLoss,
					pState->_nnfeat._sepInHiddenLoss, true);
			word_position = pState->_wordnum - 1;
			if (position < length) {
				unconcat(denseFeature._wordRNNHiddenLoss[word_position], denseFeature._actionRNNHiddenLoss[position], charLeftRNNHidden_Loss[position],
						charRightRNNHidden_Loss[position], denseFeature._postagRNNHiddenLoss[word_position], pState->_nnfeat._sepInHiddenLoss);
			} else {
				unconcat(denseFeature._wordRNNHiddenLoss[word_position], denseFeature._actionRNNHiddenLoss[position], charRNNHiddenDummy_Loss, charRNNHiddenDummy_Loss,
						denseFeature._postagRNNHiddenLoss[word_position], pState->_nnfeat._sepInHiddenLoss);
			}

		} else {
			_nnlayer_app_output.ComputeBackwardLoss(pState->_nnfeat._appOutHidden, loss, pState->_nnfeat._appOutHiddenLoss, true);
			_nnlayer_app_hidden.ComputeBackwardLoss(pState->_nnfeat._appInHidden, pState->_nnfeat._appOutHidden, pState->_nnfeat._appOutHiddenLoss,
					pState->_nnfeat._appInHiddenLoss, true);

			unconcat(denseFeature._actionRNNHiddenLoss[position], charLeftRNNHidden_Loss[position], charRightRNNHidden_Loss[position], pState->_nnfeat._appInHiddenLoss);
		}

	}

	void copyDynamicRNNValues(const CStateItem *pState, DenseFeature& denseFeature) {
		static int position, word_position;

		position = pState->_nextPosition - 1;
		word_position = pState->_wordnum - 1;

		if (pState->_lastAction.isSeparate() || pState->_lastAction.isFinish()) {

			Copy(denseFeature._allwordPrime[word_position], pState->_nnfeat._allwordPrime);
			Copy(denseFeature._allwordRep[word_position], pState->_nnfeat._allwordRep);
			Copy(denseFeature._wordUnitRep[word_position], pState->_nnfeat._wordUnitRep);
			Copy(denseFeature._wordHidden[word_position], pState->_nnfeat._wordHidden);
			for (int idk = 0; idk < _buffer; idk++) {
				Copy(denseFeature._wordRNNHiddenBuf[idk][word_position], pState->_nnfeat._wordRNNHiddenBuf[idk]);
			}
			Copy(denseFeature._wordRNNHidden[word_position], pState->_nnfeat._wordRNNHidden);
			Copy(denseFeature._wordUnitRepMask[word_position], pState->_nnfeat._wordUnitRepMask);

			for (int tmp_k = 0; tmp_k < _wordNgram; tmp_k++) {
				denseFeature._allwordIds[word_position][tmp_k] = pState->_curFeat._nAllWordFeat[tmp_k];
			}

			Copy(denseFeature._postagPrime[word_position], pState->_nnfeat._postagPrime);
			Copy(denseFeature._postagRep[word_position], pState->_nnfeat._postagRep);
			Copy(denseFeature._postagRepMask[word_position], pState->_nnfeat._postagRepMask);
			Copy(denseFeature._postagHidden[word_position], pState->_nnfeat._postagHidden);
			for (int idk = 0; idk < _buffer; idk++) {
				Copy(denseFeature._postagRNNHiddenBuf[idk][word_position], pState->_nnfeat._postagRNNHiddenBuf[idk]);
			}
			Copy(denseFeature._postagRNNHidden[word_position], pState->_nnfeat._postagRNNHidden);
			for (int tmp_k = 0; tmp_k < _postagNgram; tmp_k++) {
				denseFeature._postagIds[word_position][tmp_k] = pState->_curFeat._nPostagFeat[tmp_k];
			}

		}

		Copy(denseFeature._actionPrime[position], pState->_nnfeat._actionPrime);
		Copy(denseFeature._actionRep[position], pState->_nnfeat._actionRep);
		Copy(denseFeature._actionRepMask[position], pState->_nnfeat._actionRepMask);
		Copy(denseFeature._actionHidden[position], pState->_nnfeat._actionHidden);
		for (int idk = 0; idk < _buffer; idk++) {
			Copy(denseFeature._actionRNNHiddenBuf[idk][word_position], pState->_nnfeat._actionRNNHiddenBuf[idk]);
		}
		Copy(denseFeature._actionRNNHidden[position], pState->_nnfeat._actionRNNHidden);
		for (int tmp_k = 0; tmp_k < _actionNgram; tmp_k++) {
			denseFeature._actionIds[position][tmp_k] = pState->_curFeat._nActionFeat[tmp_k];
		}

	}

};

#endif /* SRC_LSTMBeamSearcher_H_ */
