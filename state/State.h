/*
 * State.h
 *
 *  Created on: Oct 1, 2015
 *      Author: mszhang
 */

#ifndef SEG_STATE_H_
#define SEG_STATE_H_

#include "Feature.h"
#include "DenseFeatureForward.h"
#include "Action.h"
#include "Instance.h"
#include "Alphabet.h"

#include "CTagConstraints.h"

class CStateItem {
public:
	std::string _strLastWord;
	std::string _strLastPostag;
	int _lastWordStart;
	int _lastWordEnd;
	const CStateItem *_prevStackState;
	const CStateItem *_prevSepState;
	const CStateItem *_prevState;
	int _nextPosition;

	const std::vector<std::string> *_pCharacters;
	const std::vector<std::vector<std::string> > *_pCandidateLabels;

	int _characterSize;

	CAction _lastAction;
	Feature _curFeat;
	DenseFeatureForward _nnfeat;
	dtype _score;
	int _wordnum;

public:
	CStateItem() {
		_strLastWord = "";
		_strLastPostag = "";
		_lastWordStart = -1;
		_lastWordEnd = -1;
		_prevStackState = 0;
		_prevSepState = 0;
		_prevState = 0;
		_nextPosition = 0;
		_pCharacters = 0;
		_pCandidateLabels = 0;
		_characterSize = 0;
		_lastAction.clear();
		_curFeat.clear();
		_nnfeat.clear();
		_score = 0.0;
		_wordnum = 0;
	}

	CStateItem(const std::vector<std::string>* pCharacters, const std::vector<std::vector<std::string> >* pCandidateLabels) {
		_strLastWord = "";
		_strLastPostag = "";
		_lastWordStart = -1;
		_lastWordEnd = -1;
		_prevStackState = 0;
		_prevSepState = 0;
		_prevState = 0;
		_nextPosition = 0;
		_pCharacters = pCharacters;
		_pCandidateLabels = pCandidateLabels;
		_characterSize = pCharacters->size();
		_lastAction.clear();
		_curFeat.clear();
		_nnfeat.clear();
		_score = 0.0;
		_wordnum = 0;
	}

	virtual ~CStateItem() {
		clear();
	}

	void initSentence(const std::vector<std::string>* pCharacters, const std::vector<std::vector<std::string> >* pCandidateLabels) {
		_pCharacters = pCharacters;
		_pCandidateLabels = pCandidateLabels;
		_characterSize = pCharacters->size();

	}

	void clear() {
		_strLastWord = "";
		_strLastPostag = "";
		_lastWordStart = -1;
		_lastWordEnd = -1;
		_prevStackState = 0;
		_prevSepState = 0;
		_prevState = 0;
		_nextPosition = 0;
		_lastAction.clear();
		_curFeat.clear();
		_nnfeat.clear();
		_score = 0.0;
		_wordnum = 0;
	}

	void copyState(const CStateItem* from) {
		_strLastWord = from->_strLastWord;
		_strLastPostag = from->_strLastPostag;
		_lastWordStart = from->_lastWordStart;
		_lastWordEnd = from->_lastWordEnd;
		_prevStackState = from->_prevStackState;
		_prevSepState = from->_prevSepState;
		_prevState = from->_prevState;
		_nextPosition = from->_nextPosition;
		_pCharacters = from->_pCharacters;
		_pCandidateLabels = from->_pCandidateLabels;
		_characterSize = from->_characterSize;
		_lastAction = from->_lastAction;
		_curFeat.copy(from->_curFeat);
		_nnfeat.copy(from->_nnfeat);
		_score = from->_score;
		_wordnum = from->_wordnum;
	}

	const CStateItem* getPrevStackState() const {
		return _prevStackState;
	}

	const CStateItem* getPrevSepState() const {
		return _prevSepState;
	}

	const CStateItem* getPrevState() const {
		return _prevState;
	}

	std::string getLastWord() {
		return _strLastWord;
	}

	std::string getLastPostag() {
		return _strLastPostag;
	}

public:
	//only assign context
	void separate(CStateItem* next, int posId, Alphabet& postagAlphabet) const {
		if (_nextPosition >= _characterSize) {
			std::cout << "separate error" << std::endl;
			return;
		}
		next->_strLastWord = (*_pCharacters)[_nextPosition];
		next->_strLastPostag = postagAlphabet.from_id(posId);
		next->_lastWordStart = _nextPosition;
		next->_lastWordEnd = _nextPosition;
		next->_prevStackState = this;
		next->_prevSepState = next;
		next->_prevState = this;
		next->_nextPosition = _nextPosition + 1;
		next->_pCharacters = _pCharacters;
		next->_pCandidateLabels = _pCandidateLabels;
		next->_characterSize = _characterSize;
		next->_wordnum = _wordnum + 1;
		next->_lastAction.set(CAction::SEP + posId);
	}

	//only assign context
	void finish(CStateItem* next) const {
		if (_nextPosition != _characterSize) {
			std::cout << "finish error" << std::endl;
			return;
		}
		next->_strLastWord = "";
		next->_strLastPostag = "";
		next->_lastWordStart = _nextPosition;
		next->_lastWordEnd = _nextPosition;
		next->_prevStackState = this;
		next->_prevSepState = next;
		next->_prevState = this;
		next->_nextPosition = _nextPosition + 1;
		next->_pCharacters = _pCharacters;
		next->_pCandidateLabels = _pCandidateLabels;
		next->_characterSize = _characterSize;
		next->_wordnum = _wordnum + 1;
		next->_lastAction.set(CAction::FIN);
	}

	//only assign context
	void append(CStateItem* next) const {
		if (_nextPosition >= _characterSize) {
			std::cout << "append error" << std::endl;
			return;
		}
		next->_strLastWord = _strLastWord + (*_pCharacters)[_nextPosition];
		next->_strLastPostag = _strLastPostag;
		next->_lastWordStart = _lastWordStart;
		next->_lastWordEnd = _nextPosition;
		next->_prevStackState = _prevStackState;
		next->_prevSepState = _prevSepState;
		next->_prevState = this;
		next->_nextPosition = _nextPosition + 1;
		next->_pCharacters = _pCharacters;
		next->_pCandidateLabels = _pCandidateLabels;
		next->_characterSize = _characterSize;
		next->_wordnum = _wordnum;
		next->_lastAction.set(CAction::APP);
	}

	void move(CStateItem* next, const CAction& ac, Alphabet& postagAlphabet) const {
		if (ac.isAppend()) {
			append(next);
		} else if (ac.isSeparate()) {
			separate(next, ac.getPosId(), postagAlphabet);
		} else if (ac.isFinish()) {
			finish(next);
		} else {
			std::cout << "error action" << std::endl;
		}
	}

	bool IsTerminated() const {
		if (_lastAction.isFinish())
			return true;
		return false;
	}

	//partial results
	void getSegPosResults(CResult& curResult) const {
		curResult.clear();
		if (!IsTerminated()) {
			curResult.words.push_back(_strLastWord);
			curResult.postags.push_back(_strLastPostag);
		}
		else {
			// faked, no use
		}
		const CStateItem *prevStackState = _prevStackState;
		while (prevStackState != 0 && prevStackState->_wordnum > 0) {
			curResult.words.insert(curResult.words.begin(), prevStackState->_strLastWord);
			curResult.postags.insert(curResult.postags.begin(), prevStackState->_strLastPostag);
			prevStackState = prevStackState->_prevStackState;
		}
	}

	void getGoldAction(const Instance& inst, Alphabet& postagAlphabet, CAction& ac) const {
		const std::vector<std::string> segments = inst.words;
		const std::vector<std::string> postags = inst.postags;
		if (_nextPosition == _characterSize) {
			ac.set(CAction::FIN);
			return;
		}
		if (_nextPosition == 0) {
			ac.set(CAction::SEP + postagAlphabet.from_string(postags[0]));
			return;
		}

		if (_nextPosition > 0 && _nextPosition < _characterSize) {
			// should have a check here to see whether the words are match, but I did not do it here
			if (_strLastWord.length() == segments[_wordnum - 1].length()) {
				ac.set(CAction::SEP + postagAlphabet.from_string(postags[_wordnum]));
				return;
			} else {
				ac.set(CAction::APP);
				return;
			}
		}

		ac.set(CAction::NO_ACTION);
		return;
	}

	// we did not judge whether history actions are match with current state.
	void getGoldAction(const CStateItem* goldState, CAction& ac) const {
		if (_nextPosition > goldState->_nextPosition || _nextPosition < 0) {
			ac.set(CAction::NO_ACTION);
			return;
		}
		const CStateItem *prevState = goldState->_prevState;
		CAction curAction = goldState->_lastAction;
		while (_nextPosition < prevState->_nextPosition) {
			curAction = prevState->_lastAction;
			prevState = prevState->_prevState;
		}
		return ac.set(curAction._code);
	}

	void getCandidateActions(vector<CAction> & actions, Alphabet& postagAlphabet) const {
		static string curLabel;
		static hash_set<string>::iterator hashsetIt;
		static int pos_id;
		actions.clear();
		static CAction ac;
		if (_nextPosition == _characterSize) {
			ac.set(CAction::FIN);
			actions.push_back(ac);
		}
		else if (_nextPosition >= 0 && _nextPosition < _characterSize) {
			/*
			 for (int pos_id = 1; pos_id < postagAlphabet.size(); pos_id++) {
			 ac.set(CAction::SEP + pos_id);
			 actions.push_back(ac);
			 }

			 if (_nextPosition > 0) {
			 ac.set(CAction::APP);
			 actions.push_back(ac);
			 }
			 */
            
			hash_set<string> candidateTags;
			candidateTags.clear();
			bool bCanAppend = false;
			for (int idx = 0; idx < (*_pCandidateLabels)[_nextPosition].size(); idx++) {
				curLabel = (*_pCandidateLabels)[_nextPosition][idx];
				if (curLabel.length() > 2 && curLabel[0] == 'b' && curLabel[1] == '-') {
					candidateTags.insert(curLabel.substr(2));
				}
				else if (curLabel.length() == 1 && curLabel[0] == 'i') {
					bCanAppend = true;
				}
				else {
					std::cout << "error candidate label: " << curLabel << std::endl;
				}
			}

			for (hashsetIt = candidateTags.begin(); hashsetIt != candidateTags.end(); hashsetIt++) {
				pos_id = postagAlphabet.from_string(*hashsetIt);
				if (pos_id > 0) {
					ac.set(CAction::SEP + pos_id);
					actions.push_back(ac);
				}
			}
			if (_nextPosition > 0 && bCanAppend) {
				ac.set(CAction::APP);
				actions.push_back(ac);
			}
			
			if(actions.size() == 0){
				for (int pos_id = 1; pos_id < postagAlphabet.size(); pos_id++) {
					ac.set(CAction::SEP + pos_id);
					actions.push_back(ac);
				}
				
				if (_nextPosition > 0) {
					ac.set(CAction::APP);
					actions.push_back(ac);
				}				
			}

		}
		else {
			std::cout << "impossible" << std::endl;
		}

	}

	inline std::string str() const {
		stringstream curoutstr;

		curoutstr << "score: " << _score << std::endl;
		curoutstr << "result:";
		CResult curResult;
		getSegPosResults(curResult);
		for (int idx = 0; idx < curResult.size(); idx++) {
			curoutstr << " " << curResult.words[idx] << "_" << curResult.postags[idx];
		}
		curoutstr << std::endl;
		curoutstr << "action: ";
		vector<string> actions;
		actions.push_back(this->_lastAction.str());
		const CStateItem *prevState = this->_prevState;
		while (prevState != 0) {
			actions.insert(actions.begin(), prevState->_lastAction.str());
			prevState = prevState->_prevState;
		}

		for (int idx = 0; idx < actions.size(); idx++) {
			curoutstr << " " << actions[idx];
		}

		return curoutstr.str();
	}

};

class CScoredStateAction {
public:
	CAction action;
	const CStateItem *item;
	dtype score;
	Feature feat;
	DenseFeatureForward nnfeat;

public:
	CScoredStateAction() :
			item(0), action(-1), score(0) {
		feat.setFeatureFormat(false);
		feat.clear();
		nnfeat.clear();
	}

public:
	inline CScoredStateAction& operator=(const CScoredStateAction &rhs) {
		// Check for self-assignment!
		if (this == &rhs)      // Same object?
			return *this;        // Yes, so skip assignment, and just return *this.

		item = rhs.item;
		action.set(rhs.action._code);
		score = rhs.score;
		feat.copy(rhs.feat);
		nnfeat.copy(rhs.nnfeat);

		return *this;
	}

public:
	bool operator <(const CScoredStateAction &a1) const {
		return score < a1.score;
	}
	bool operator >(const CScoredStateAction &a1) const {
		return score > a1.score;
	}
	bool operator <=(const CScoredStateAction &a1) const {
		return score <= a1.score;
	}
	bool operator >=(const CScoredStateAction &a1) const {
		return score >= a1.score;
	}

};

class CScoredStateAction_Compare {
public:
	int operator()(const CScoredStateAction &o1, const CScoredStateAction &o2) const {

		if (o1.score < o2.score)
			return -1;
		else if (o1.score > o2.score)
			return 1;
		else
			return 0;
	}
};

#endif /* SEG_STATE_H_ */
