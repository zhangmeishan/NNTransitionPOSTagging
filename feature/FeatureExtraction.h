/*
 * FeatureExtraction.h
 *
 *  Created on: Oct 7, 2015
 *      Author: mszhang
 */

#ifndef BASIC_FEATUREEXTRACTION_H_
#define BASIC_FEATUREEXTRACTION_H_
#include "N3L.h"
#include "Action.h"
#include "State.h"
#include "Utf.h"

class FeatureExtraction {
public:
	std::string nullkey;
	std::string rootdepkey;
	std::string unknownkey;
	std::string paddingtag;
	std::string seperateKey;

public:
	Alphabet _featAlphabet;
	Alphabet _wordAlphabet;
	Alphabet _allwordAlphabet;
	Alphabet _charAlphabet;
	Alphabet _bicharAlphabet;
	Alphabet _actionAlphabet;
	Alphabet _postagAlphabet;

public:
	bool _bStringFeat; // string-formated features or digit-formated features
  CTagConstraints _tagConstraints;
  int _seed;

public:
	FeatureExtraction() {
		nullkey = "-null-";
		unknownkey = "-unknown-";
		paddingtag = "-padding-";
		seperateKey = "#";

		_bStringFeat = true;
		_seed = 0;
	}

	FeatureExtraction(bool bCollecting) {
		nullkey = "-null-";
		unknownkey = "-unknown-";
		paddingtag = "-padding-";
		seperateKey = "#";

		_bStringFeat = bCollecting;
		_seed = 0;
	}

public:

	inline void setFeatureFormat(bool bStringFeat) {
		_bStringFeat = bStringFeat;
	}

	inline void setAlphaIncreasing(bool alphaIncreasing) {
		if (alphaIncreasing) {
			_featAlphabet.set_fixed_flag(false);
			_wordAlphabet.set_fixed_flag(false);
			_allwordAlphabet.set_fixed_flag(false);
			_charAlphabet.set_fixed_flag(false);
			_bicharAlphabet.set_fixed_flag(false);
			_actionAlphabet.set_fixed_flag(false);
			_postagAlphabet.set_fixed_flag(false);
		} else {
			_featAlphabet.set_fixed_flag(true);
			_wordAlphabet.set_fixed_flag(true);
			_allwordAlphabet.set_fixed_flag(true);
			_charAlphabet.set_fixed_flag(true);
			_bicharAlphabet.set_fixed_flag(true);
			_actionAlphabet.set_fixed_flag(true);
			_postagAlphabet.set_fixed_flag(true);
		}
	}

	inline void setFeatAlphaIncreasing(bool alphaIncreasing) {
		if (alphaIncreasing) {
			_featAlphabet.set_fixed_flag(false);
		} else {
			_featAlphabet.set_fixed_flag(true);
		}
	}

	inline int getCharAlphaId(const std::string & oneChar) {
		return _charAlphabet[oneChar];
	}

	inline int getBiCharAlphaId(const std::string & twoChar) {
		return _bicharAlphabet[twoChar];
	}

	void extractFeature(const CStateItem * curState, const CAction& nextAC, Feature& feat, int wordNgram = 0, int postagNgram = 0, int actionNgram = 0, dtype dropOut = 0.0) {
		feat.clear();
		feat.setFeatureFormat(_bStringFeat);
		if (nextAC._code == CAction::APP) {
			extractFeatureApp(curState, feat, wordNgram, postagNgram, actionNgram);
		} else if (nextAC._code >= CAction::SEP) {
			extractFeatureSep(curState, nextAC, feat, wordNgram, postagNgram, actionNgram);
		} else if (nextAC._code == CAction::FIN) {
			extractFeatureFinish(curState, nextAC, feat, wordNgram, postagNgram, actionNgram);
		} else {

		}

		static int featId, unknownID;
		if (!_bStringFeat) {
			for (int idx = 0; idx < feat._strSparseFeat.size(); idx++) {
				featId = _featAlphabet[feat._strSparseFeat[idx]];
				if (featId >= 0)
					feat._nSparseFeat.push_back(featId);
			}
			feat._strSparseFeat.clear();

			if (wordNgram > 0) {
				feat._nWordFeat.resize(wordNgram);
				unknownID = _wordAlphabet[unknownkey];
				for (int idx = 0; idx < wordNgram; idx++) {
					featId = idx < feat._strWordFeat.size() ? _wordAlphabet[normalize_to_lowerwithdigit(feat._strWordFeat[idx])] : _wordAlphabet[nullkey];
					if (featId < 0)
						featId = unknownID;
					feat._nWordFeat[idx] = featId;
				}

				feat._nAllWordFeat.resize(wordNgram);
				unknownID = _allwordAlphabet[unknownkey];
				for (int idx = 0; idx < wordNgram; idx++) {
					featId = idx < feat._strWordFeat.size() ? _allwordAlphabet[feat._strWordFeat[idx]] : _allwordAlphabet[nullkey];
					if (featId < 0)
						featId = unknownID;
					feat._nAllWordFeat[idx] = featId;
				}

				feat._strWordFeat.clear();

				for(int idx = feat._nWordLengths.size(); idx < wordNgram; idx++){
					feat._nWordLengths.push_back(0);
				}

				feat._nKeyChars.resize(2 * wordNgram + 1);
				for (int idx = 0; idx < 2 * wordNgram + 1; idx++) {
					featId = idx < feat._strKeyChars.size() ? _charAlphabet[feat._strKeyChars[idx]] : _charAlphabet[nullkey];
					if (featId < 0)
						featId = unknownID;
					feat._nKeyChars[idx] = featId;
				}

			}

			if (postagNgram > 0) {
				feat._nPostagFeat.resize(postagNgram);
				for (int idx = 0; idx < postagNgram; idx++) {
					featId = idx < feat._strPostagFeat.size() ? _postagAlphabet[feat._strPostagFeat[idx]] : _postagAlphabet[nullkey];
					if(featId == -1) featId = 0; //noPostag
					feat._nPostagFeat[idx] = featId;
				}
				feat._strPostagFeat.clear();
			}

			if (actionNgram > 0) {
				feat._nActionFeat.resize(actionNgram);
				for (int idx = 0; idx < actionNgram; idx++) {
					featId = idx < feat._strActionFeat.size() ? _actionAlphabet[feat._strActionFeat[idx]] : _actionAlphabet[nullkey];
					if(featId == -1) featId = 0; //noAction
					feat._nActionFeat[idx] = featId;
				}
				feat._strActionFeat.clear();
			}
		}


	}

	void addToFeatureAlphabet(hash_map<string, int> feat_stat, int featCutOff = 0) {
		_featAlphabet.set_fixed_flag(false);
		hash_map<string, int>::iterator feat_iter;
		for (feat_iter = feat_stat.begin(); feat_iter != feat_stat.end(); feat_iter++) {
			if (feat_iter->second > featCutOff) {
				_featAlphabet.from_string(feat_iter->first);
			}
		}
		_featAlphabet.set_fixed_flag(true);
	}

	void addToWordAlphabet(hash_map<string, int> word_stat, int wordCutOff = 0) {
		_wordAlphabet.set_fixed_flag(false);
		hash_map<string, int>::iterator word_iter;
		for (word_iter = word_stat.begin(); word_iter != word_stat.end(); word_iter++) {
			if (word_iter->second > wordCutOff) {
				_wordAlphabet.from_string(word_iter->first);
			}
		}
		_wordAlphabet.set_fixed_flag(true);
	}

	void addToAllWordAlphabet(hash_map<string, int> allword_stat, int allwordCutOff = 0) {
		_allwordAlphabet.set_fixed_flag(false);
		hash_map<string, int>::iterator allword_iter;
		for (allword_iter = allword_stat.begin(); allword_iter != allword_stat.end(); allword_iter++) {
			if (allword_iter->second > allwordCutOff) {
				_allwordAlphabet.from_string(allword_iter->first);
			}
		}
		_allwordAlphabet.set_fixed_flag(true);
	}

	void addToCharAlphabet(hash_map<string, int> char_stat, int charCutOff = 0) {
		_charAlphabet.set_fixed_flag(false);
		hash_map<string, int>::iterator char_iter;
		for (char_iter = char_stat.begin(); char_iter != char_stat.end(); char_iter++) {
			if (char_iter->second > charCutOff) {
				_charAlphabet.from_string(char_iter->first);
			}
		}
		_charAlphabet.set_fixed_flag(true);
	}

	void addToBiCharAlphabet(hash_map<string, int> bichar_stat, int bicharCutOff = 0) {
		_bicharAlphabet.set_fixed_flag(false);
		hash_map<string, int>::iterator bichar_iter;
		for (bichar_iter = bichar_stat.begin(); bichar_iter != bichar_stat.end(); bichar_iter++) {
			if (bichar_iter->second > bicharCutOff) {
				_bicharAlphabet.from_string(bichar_iter->first);
			}
		}
		_bicharAlphabet.set_fixed_flag(true);
	}

	void addToActionAlphabet(hash_map<string, int> action_stat) {
		_actionAlphabet.set_fixed_flag(false);
		hash_map<string, int>::iterator action_iter;
		for (action_iter = action_stat.begin(); action_iter != action_stat.end(); action_iter++) {
			_actionAlphabet.from_string(action_iter->first);
		}
		_actionAlphabet.set_fixed_flag(true);
	}

	void addToPostagAlphabet(hash_map<string, int> pos_stat) {
		_postagAlphabet.set_fixed_flag(false);
		hash_map<string, int>::iterator pos_iter;
		for (pos_iter = pos_stat.begin(); pos_iter != pos_stat.end(); pos_iter++) {
			_postagAlphabet.from_string(pos_iter->first);
		}
		_postagAlphabet.set_fixed_flag(true);
	}

	void initAlphabet() {
		//alphabet initialization
		_featAlphabet.clear();
		_featAlphabet.set_fixed_flag(true);

		_wordAlphabet.clear();
		_wordAlphabet.from_string(nullkey);
		_wordAlphabet.from_string(unknownkey);
		_wordAlphabet.set_fixed_flag(true);

		_allwordAlphabet.clear();
		_allwordAlphabet.from_string(nullkey);
		_allwordAlphabet.from_string(unknownkey);
		_allwordAlphabet.set_fixed_flag(true);

		_charAlphabet.clear();
		_charAlphabet.from_string(nullkey);
		_charAlphabet.from_string(unknownkey);
		_charAlphabet.set_fixed_flag(true);

		_bicharAlphabet.clear();
		_bicharAlphabet.from_string(nullkey);
		_bicharAlphabet.from_string(unknownkey);
		_bicharAlphabet.set_fixed_flag(true);

		_postagAlphabet.clear();
		_postagAlphabet.from_string(nullkey);
		_postagAlphabet.set_fixed_flag(true);

		_actionAlphabet.clear();
		_actionAlphabet.from_string(nullkey);
		_actionAlphabet.set_fixed_flag(true);

		_bStringFeat = true;
	}

	void loadAlphabet() {
		_bStringFeat = false;
	}

protected:
	void extractFeatureApp(const CStateItem * curState, Feature& feat, int wordNgram = 0, int postagNgram = 0, int actionNgram = 0) {
		string curWord = curState->_lastWordEnd == -1 ? nullkey : curState->_strLastWord;
		const std::vector<std::string> * pCharacters = curState->_pCharacters;
		string nextChar = pCharacters->at(curState->_nextPosition);
		string curWordLastChar = curState->_lastWordEnd == -1 ? nullkey : pCharacters->at(curState->_lastWordEnd);
		string curWordLast2Char = curState->_lastWordEnd < 1 ? nullkey : pCharacters->at(curState->_lastWordEnd - 1);
		string curWordFirstChar = curState->_lastWordEnd == -1 ? nullkey : pCharacters->at(curState->_lastWordStart);

		string curPostag = curState->_strLastPostag;

		string curWordLastCharType = curState->_lastWordEnd == -1 ? nullkey : wordtype(curWordLastChar);
		string curWordLast2CharType = curState->_lastWordEnd < 1 ? nullkey : wordtype(curWordLast2Char);
		string nextCharType = wordtype(nextChar);

		if (curState->_nextPosition != curState->_lastWordEnd + 1) {
			std::cout << "position error" << std::endl;
		}

		string strFeat = "";
		strFeat = "F01" + seperateKey + curWordLastChar + seperateKey + nextChar;			//c-1,c0    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F02" + seperateKey + curWordFirstChar + seperateKey + nextChar;		//start(w0),c0
		feat._strSparseFeat.push_back(strFeat);

		/*
		 strFeat = "F03" + seperateKey + nextCharType;
		 feat._strSparseFeat.push_back(strFeat);

		 strFeat = "F04" + seperateKey + curWordLastCharType + nextCharType;
		 feat._strSparseFeat.push_back(strFeat);

		 */
		strFeat = "F05" + seperateKey + curWordLast2CharType + curWordLastCharType + nextCharType;
		feat._strSparseFeat.push_back(strFeat);

		//postag
		strFeat = "F06" + seperateKey + curPostag + seperateKey + nextChar;		//t0,c0    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F07" + seperateKey + nextChar + seperateKey + curPostag + seperateKey + curWordFirstChar;		//c0,t0,start(w0)    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F08" + seperateKey + curWordLastChar + seperateKey + curPostag + seperateKey + nextChar;		//c-1,t0,c0    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		int ngram = 0;

		if (ngram < actionNgram) {
			feat._strActionFeat.push_back(CAction(CAction::APP).str());
			ngram++;
			const CStateItem* theState = curState;
			while (theState != NULL && ngram < actionNgram) {
				feat._strActionFeat.push_back(theState->_lastAction.str());
				theState = theState->_prevState;
				ngram++;
			}
		}
	}

	void extractFeatureSep(const CStateItem * curState, const CAction& ac, Feature& feat, int wordNgram = 0, int postagNgram = 0, int actionNgram = 0) {
		string curWord = curState->_lastWordEnd == -1 ? nullkey : curState->_strLastWord;
		const std::vector<std::string> * pCharacters = curState->_pCharacters;
		string nextChar = pCharacters->at(curState->_nextPosition);
		string curWordLastChar = curState->_lastWordEnd == -1 ? nullkey : pCharacters->at(curState->_lastWordEnd);
		string curWordLast2Char = curState->_lastWordEnd < 1 ? nullkey : pCharacters->at(curState->_lastWordEnd - 1);
		string curWordFirstChar = curState->_lastWordEnd == -1 ? nullkey : pCharacters->at(curState->_lastWordStart);

		string curPostag = curState->_lastWordEnd == -1 ? nullkey : curState->_strLastPostag;
		string nextPostag = _postagAlphabet.from_id(ac.getPosId());

		string curWordLastCharType = curState->_lastWordEnd == -1 ? nullkey : wordtype(curWordLastChar);
		string curWordLast2CharType = curState->_lastWordEnd < 1 ? nullkey : wordtype(curWordLast2Char);
		string nextCharType = wordtype(nextChar);

		int length = curState->_lastWordEnd - curState->_lastWordStart + 1;
		if (length > 5)
			length = 5;
		stringstream curss;
		curss << length;
		string strCurWordLen = curss.str();

		if (curState->_nextPosition != curState->_lastWordEnd + 1) {
			std::cout << "position error" << std::endl;
		}

		const CStateItem * preStackState = curState->_prevStackState;
		string pre1Word = preStackState == 0 || preStackState->_lastWordEnd == -1 ? nullkey : preStackState->_strLastWord;
		string pre1WordLastChar = preStackState == 0 || preStackState->_lastWordEnd == -1 ? nullkey : pCharacters->at(preStackState->_lastWordEnd);
		string pre1WordFirstChar = preStackState == 0 || preStackState->_lastWordEnd == -1 ? nullkey : pCharacters->at(preStackState->_lastWordStart);

		string pre1Postag = preStackState == 0 || preStackState->_lastWordEnd == -1 ? nullkey : preStackState->_strLastPostag;

		length = preStackState == 0 ? 0 : preStackState->_lastWordEnd - preStackState->_lastWordStart + 1;
		if (length > 5)
			length = 5;
		stringstream press;
		press << length;
		string strPreWordLen = press.str();

		string strFeat = "";
		strFeat = "F11" + seperateKey + curWordLastChar + seperateKey + nextChar;			//end(w-1),c0    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F12" + seperateKey + curWordFirstChar + seperateKey + nextChar;		//start(w-1),c0    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		/*
		 strFeat = "F13" + seperateKey + nextCharType;
		 feat._strSparseFeat.push_back(strFeat);

		 strFeat = "F14" + seperateKey + curWordLastCharType + nextCharType;
		 feat._strSparseFeat.push_back(strFeat);
		 */
		strFeat = "F15" + seperateKey + curWordLast2CharType + curWordLastCharType + nextCharType;
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F16" + seperateKey + curWord + seperateKey + nextChar;			//w-1,c0    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F17" + seperateKey + curWord;															//w-1    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F18" + seperateKey + curWord + seperateKey + pre1Word;			//w-1,w-2    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F19" + seperateKey + curWord + seperateKey + pre1WordLastChar;						//w-1,end(w-2)    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F20" + seperateKey + curWord + seperateKey + pre1WordFirstChar;					//w-1,start(w-2)
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F21" + seperateKey + curWordLastChar + seperateKey + pre1WordLastChar;		//end(w-1),end(w-2)    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F22" + seperateKey + curWordFirstChar + seperateKey + curWordLastChar;		//star(w-1),end(w-1)
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F23" + seperateKey + curWord + seperateKey + strPreWordLen;							//w-1,len(w-2)    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F24" + seperateKey + pre1Word + seperateKey + strCurWordLen;							//w-2,len(w-1)    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F25" + seperateKey + pre1Word + seperateKey + curWordLastChar;						//w-2,end(w-1)
		feat._strSparseFeat.push_back(strFeat);

		if (curState->_lastWordStart != -1) {
			for (int idx = curState->_lastWordStart; idx < curState->_lastWordEnd; idx++) {
				strFeat = "F26" + seperateKey + pCharacters->at(idx) + seperateKey + curWordLastChar;						//eachCharacterIn(w-1)ExceptEnd(w-1),end(w-1)
				feat._strSparseFeat.push_back(strFeat);
			}
		}

		if (curState->_lastWordEnd == curState->_lastWordStart && curState->_lastWordStart != -1) {
			strFeat = "F27" + seperateKey + curWord;																													//w-1, where len(w-1) = 1    Z&C10
			feat._strSparseFeat.push_back(strFeat);
		}

		strFeat = "F28" + seperateKey + curWordFirstChar + seperateKey + strCurWordLen;											//start(w-1),len(w-1)    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F29" + seperateKey + curWordLastChar + seperateKey + strCurWordLen;											//end(w-1),len(w-1)    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		//postag
		strFeat = "F30" + seperateKey + curWord + seperateKey + curPostag;															//w-1,t-1    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F31" + seperateKey + curPostag + seperateKey + nextPostag;												//t-1,t0    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F32" + seperateKey + pre1Postag + seperateKey + curPostag + seperateKey + nextPostag;		//t-2,t-1,t0    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F33" + seperateKey + curWord + seperateKey + nextPostag;															//w-1,t0    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F34" + seperateKey + pre1Postag + seperateKey + curWord;															//t-2,w-1    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F35" + seperateKey + curWord + seperateKey + curPostag + seperateKey + pre1WordLastChar;					//w-1,t-1,end(w-2)    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F36" + seperateKey + curWord + seperateKey + curPostag + seperateKey + nextChar;									//w-1,t-1,c0    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		//c-2,c-1,c0,t-1 where len(w-1) = 1   Z&C10
		if (curState->_lastWordEnd == curState->_lastWordStart && curState->_lastWordStart != -1) {
			strFeat = "F37" + seperateKey + curWordLast2Char + seperateKey + curWordLastChar + seperateKey + nextChar + seperateKey + curPostag;
			feat._strSparseFeat.push_back(strFeat);
		}

		strFeat = "F38" + seperateKey + nextChar + seperateKey + nextPostag;															//start(w0),t0    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F39" + seperateKey + curPostag + seperateKey + curWordFirstChar;												//t-1,start(w-1)    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		//c0,t0,c-1,t-1)    Z&C10
		strFeat = "F40" + seperateKey + nextChar + seperateKey + nextPostag + seperateKey + curWordLastChar + seperateKey + curPostag;
		feat._strSparseFeat.push_back(strFeat);

    if (_tagConstraints.word_freq.find(curWord) != _tagConstraints.word_freq.end()) {
      if (_tagConstraints.word_freq[curWord] > (_tagConstraints.maxFreqWord / 5000 + 5)) {
        strFeat = "F101";
        feat._strSparseFeat.push_back(strFeat);
        if (_tagConstraints.inWord_posMap(curWord, curPostag)) {
          strFeat = "F102";
          feat._strSparseFeat.push_back(strFeat);
        }
      }
    }

		int ngram = 0;

		if (ngram < actionNgram) {
			feat._strActionFeat.push_back(ac.str());
			ngram++;
			const CStateItem* theState = curState;
			while (theState != NULL && ngram < actionNgram) {
				feat._strActionFeat.push_back(theState->_lastAction.str());
				theState = theState->_prevState;
				ngram++;
			}
		}

		ngram = 0;

		feat._strKeyChars.push_back(nextChar);
		if (ngram < wordNgram) {
			feat._strWordFeat.push_back(curWord);
			feat._strKeyChars.push_back(curWordLastChar);
			feat._strKeyChars.push_back(curWordFirstChar);
			length = curState->_lastWordEnd == -1 ?  0 : curState->_lastWordEnd - curState->_lastWordStart + 1;
			if(length > 5) length = 5;
			feat._nWordLengths.push_back(length);
			ngram++;
			const CStateItem* theState = preStackState;
			while (theState != NULL && theState->_lastWordEnd >= 0 && ngram < wordNgram) {
				feat._strWordFeat.push_back(theState->_strLastWord);
				feat._strKeyChars.push_back(pCharacters->at(theState->_lastWordEnd));
				feat._strKeyChars.push_back(pCharacters->at(theState->_lastWordEnd));
				length = theState->_lastWordEnd - theState->_lastWordStart + 1;
				if(length > 5) length = 5;
				feat._nWordLengths.push_back(length);
				theState = theState->_prevStackState;
				ngram++;
			}
		}

		ngram = 0;

		if (ngram < postagNgram) {
			feat._strPostagFeat.push_back(curPostag);
			ngram++;
			const CStateItem* theState = preStackState;
			while (theState != NULL && theState->_lastWordEnd >= 0 && ngram < postagNgram) {
				feat._strPostagFeat.push_back(theState->_strLastPostag);
				theState = theState->_prevStackState;
				ngram++;
			}
		}

	}

	void extractFeatureFinish(const CStateItem * curState, const CAction& ac, Feature& feat, int wordNgram = 0, int postagNgram = 0, int actionNgram = 0) {
		string curWord = curState->_lastWordEnd == -1 ? nullkey : curState->_strLastWord;
		const std::vector<std::string> * pCharacters = curState->_pCharacters;
		string nextChar = nullkey;
		string curWordLastChar = curState->_lastWordEnd == -1 ? nullkey : pCharacters->at(curState->_lastWordEnd);
		string curWordLast2Char = curState->_lastWordEnd < 1 ? nullkey : pCharacters->at(curState->_lastWordEnd - 1);
		string curWordFirstChar = curState->_lastWordEnd == -1 ? nullkey : pCharacters->at(curState->_lastWordStart);

		string curPostag = curState->_lastWordEnd == -1 ? nullkey : curState->_strLastPostag;
		string nextPostag = nullkey;

		string curWordLastCharType = curState->_lastWordEnd == -1 ? nullkey : wordtype(curWordLastChar);
		string curWordLast2CharType = curState->_lastWordEnd < 1 ? nullkey : wordtype(curWordLast2Char);
		//string nextCharType = wordtype(nextChar);

		int length = curState->_lastWordEnd - curState->_lastWordStart + 1;
		if (length > 5)
			length = 5;
		stringstream curss;
		curss << length;
		string strCurWordLen = curss.str();

		if (curState->_nextPosition != curState->_lastWordEnd + 1) {
			std::cout << "position error" << std::endl;
		}

		const CStateItem * preStackState = curState->_prevStackState;
		string pre1Word = preStackState == 0 || preStackState->_lastWordEnd == -1 ? nullkey : preStackState->_strLastWord;
		string pre1WordLastChar = preStackState == 0 || preStackState->_lastWordEnd == -1 ? nullkey : pCharacters->at(preStackState->_lastWordEnd);
		string pre1WordFirstChar = preStackState == 0 || preStackState->_lastWordEnd == -1 ? nullkey : pCharacters->at(preStackState->_lastWordStart);

		string pre1Postag = preStackState == 0 || preStackState->_lastWordEnd == -1 ? nullkey : preStackState->_strLastPostag;

		length = preStackState == 0 ? 0 : preStackState->_lastWordEnd - preStackState->_lastWordStart + 1;
		if (length > 5)
			length = 5;
		stringstream press;
		press << length;
		string strPreWordLen = press.str();

		string strFeat = "";
		strFeat = "F11" + seperateKey + curWordLastChar + seperateKey + nextChar;			//end(w-1),c0    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F12" + seperateKey + curWordFirstChar + seperateKey + nextChar;		//start(w-1),c0    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		/*
		 strFeat = "F13" + seperateKey + nextCharType;
		 feat._strSparseFeat.push_back(strFeat);

		 strFeat = "F14" + seperateKey + curWordLastCharType + nextCharType;
		 feat._strSparseFeat.push_back(strFeat);
		 */
		//strFeat = "F15" + seperateKey + curWordLast2CharType + curWordLastCharType + nextCharType;
		//feat._strSparseFeat.push_back(strFeat);

		strFeat = "F16" + seperateKey + curWord + seperateKey + nextChar;			//w-1,c0    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F17" + seperateKey + curWord;															//w-1    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F18" + seperateKey + curWord + seperateKey + pre1Word;			//w-1,w-2    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F19" + seperateKey + curWord + seperateKey + pre1WordLastChar;						//w-1,end(w-2)    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F20" + seperateKey + curWord + seperateKey + pre1WordFirstChar;					//w-1,start(w-2)
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F21" + seperateKey + curWordLastChar + seperateKey + pre1WordLastChar;		//end(w-1),end(w-2)    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F22" + seperateKey + curWordFirstChar + seperateKey + curWordLastChar;		//star(w-1),end(w-1)
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F23" + seperateKey + curWord + seperateKey + strPreWordLen;							//w-1,len(w-2)    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F24" + seperateKey + pre1Word + seperateKey + strCurWordLen;							//w-2,len(w-1)    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F25" + seperateKey + pre1Word + seperateKey + curWordLastChar;						//w-2,end(w-1)
		feat._strSparseFeat.push_back(strFeat);

		if (curState->_lastWordStart != -1) {
			for (int idx = curState->_lastWordStart; idx < curState->_lastWordEnd; idx++) {
				strFeat = "F26" + seperateKey + pCharacters->at(idx) + seperateKey + curWordLastChar;						//eachCharacterIn(w-1)ExceptEnd(w-1),end(w-1)
				feat._strSparseFeat.push_back(strFeat);
			}
		}

		if (curState->_lastWordEnd == curState->_lastWordStart && curState->_lastWordStart != -1) {
			strFeat = "F27" + seperateKey + curWord;																													//w-1, where len(w-1) = 1    Z&C10
			feat._strSparseFeat.push_back(strFeat);
		}

		strFeat = "F28" + seperateKey + curWordFirstChar + seperateKey + strCurWordLen;											//start(w-1),len(w-1)    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F29" + seperateKey + curWordLastChar + seperateKey + strCurWordLen;											//end(w-1),len(w-1)    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		//postag
		strFeat = "F30" + seperateKey + curWord + seperateKey + curPostag;															//w-1,t-1    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F31" + seperateKey + curPostag + seperateKey + nextPostag;												//t-1,t0    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F32" + seperateKey + pre1Postag + seperateKey + curPostag + seperateKey + nextPostag;		//t-2,t-1,t0    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F33" + seperateKey + curWord + seperateKey + nextPostag;															//w-1,t0    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F34" + seperateKey + pre1Postag + seperateKey + curWord;															//t-2,w-1    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F35" + seperateKey + curWord + seperateKey + curPostag + seperateKey + pre1WordLastChar;					//w-1,t-1,end(w-2)    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F36" + seperateKey + curWord + seperateKey + curPostag + seperateKey + nextChar;									//w-1,t-1,c0    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		//c-2,c-1,c0,t-1 where len(w-1) = 1   Z&C10
		if (curState->_lastWordEnd == curState->_lastWordStart && curState->_lastWordStart != -1) {
			strFeat = "F37" + seperateKey + curWordLast2Char + seperateKey + curWordLastChar + seperateKey + nextChar + seperateKey + curPostag;
			feat._strSparseFeat.push_back(strFeat);
		}

		strFeat = "F38" + seperateKey + nextChar + seperateKey + nextPostag;															//start(w0),t0    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		strFeat = "F39" + seperateKey + curPostag + seperateKey + curWordFirstChar;												//t-1,start(w-1)    Z&C10
		feat._strSparseFeat.push_back(strFeat);

		//c0,t0,c-1,t-1)    Z&C10
		strFeat = "F40" + seperateKey + nextChar + seperateKey + nextPostag + seperateKey + curWordLastChar + seperateKey + curPostag;
		feat._strSparseFeat.push_back(strFeat);

    if (_tagConstraints.word_freq.find(curWord) != _tagConstraints.word_freq.end()) {
      if (_tagConstraints.word_freq[curWord] > (_tagConstraints.maxFreqWord / 5000 + 5)) {
        strFeat = "F101";
        feat._strSparseFeat.push_back(strFeat);
        if (_tagConstraints.inWord_posMap(curWord, curPostag)) {
          strFeat = "F102";
          feat._strSparseFeat.push_back(strFeat);
        }
      }
    }

		int ngram = 0;

		if (ngram < actionNgram) {
			feat._strActionFeat.push_back(ac.str());
			ngram++;
			const CStateItem* theState = curState;
			while (theState != NULL && ngram < actionNgram) {
				feat._strActionFeat.push_back(theState->_lastAction.str());
				theState = theState->_prevState;
				ngram++;
			}
		}

		ngram = 0;

		feat._strKeyChars.push_back(nullkey);
		if (ngram < wordNgram) {
			feat._strWordFeat.push_back(curWord);
			feat._strKeyChars.push_back(curWordLastChar);
			feat._strKeyChars.push_back(curWordFirstChar);
			length = curState->_lastWordEnd == -1 ?  0 : curState->_lastWordEnd - curState->_lastWordStart + 1;
			if(length > 5) length = 5;
			feat._nWordLengths.push_back(length);
			ngram++;
			const CStateItem* theState = preStackState;
			while (theState != NULL && theState->_lastWordEnd >= 0 && ngram < wordNgram) {
				feat._strWordFeat.push_back(theState->_strLastWord);
				feat._strKeyChars.push_back(pCharacters->at(theState->_lastWordEnd));
				feat._strKeyChars.push_back(pCharacters->at(theState->_lastWordEnd));
				length = theState->_lastWordEnd - theState->_lastWordStart + 1;
				if(length > 5) length = 5;
				feat._nWordLengths.push_back(length);
				theState = theState->_prevStackState;
				ngram++;
			}
		}

		ngram = 0;

		if (ngram < postagNgram) {
			feat._strPostagFeat.push_back(curPostag);
			ngram++;
			const CStateItem* theState = preStackState;
			while (theState != NULL && theState->_lastWordEnd >= 0 && ngram < postagNgram) {
				feat._strPostagFeat.push_back(theState->_strLastPostag);
				theState = theState->_prevStackState;
				ngram++;
			}
		}

	}

};

#endif /* BASIC_FEATUREEXTRACTION_H_ */
