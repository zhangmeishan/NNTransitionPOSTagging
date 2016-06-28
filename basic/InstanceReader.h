#ifndef _CONLL_READER_
#define _CONLL_READER_

#include "Reader.h"
#include "N3L.h"
#include "Utf.h"
#include <sstream>

using namespace std;
/*
 this class reads conll-format data (10 columns, no srl-info)
 */
class InstanceReader: public Reader {
public:
	InstanceReader() {
	}
	~InstanceReader() {
	}

	Instance *getNext() {
		m_instance.clear();
		static vector<string> vecLine, vecInfo, vecItems, charInfo;
		static string curWord, curPos, sentence;
		static bool bValid;
		vecLine.clear();
		static string strLine;

		while (1) {
			if (!my_getline(m_inf, strLine)) {
				break;
			}
			if (strLine.empty())
				break;
			vecLine.push_back(strLine);
		}

		int char_length = vecLine.size();

		if (char_length == 0) {
			return NULL;
		}

		curWord = "";
		curPos = "";
		sentence = "";
		bValid = true;
		for (int i = 0; i < char_length; ++i) {
			split_bychar(vecLine[i], vecInfo, '\t');
			if (vecInfo.size() != 3 || !checkLabel(vecInfo[1])
					|| (i == 0 && vecInfo[1].length() == 1)) {
				std::cout << "error line, please check." << std::endl;
				bValid = false;
				break;
			}

			if (vecInfo[1].length() > 2) {
				if (i > 0) {
					if (curWord.length() == 0 || curPos.length() == 0) {
						std::cout << "impossible in reading inputs" << std::endl;
						bValid = false;
						break;
					}
					m_instance.words.push_back(curWord);
					m_instance.postags.push_back(curPos);
				}
				curWord = "";
				curPos = vecInfo[1].substr(2);
			}

			curWord = curWord + vecInfo[0];
			sentence = sentence + vecInfo[0];
			m_instance.chars.push_back(vecInfo[0]);

			split_bychar(vecInfo[2], vecItems, '#');
			m_instance.candidateLabels.push_back(vecItems);
		}

		if (bValid) {
			m_instance.words.push_back(curWord);
			m_instance.postags.push_back(curPos);
		}

		getCharactersFromUTF8String(sentence, charInfo);
		if (charInfo.size() != char_length) {
			std::cout << "character length size does not equal." << std::endl;
			bValid = false;
		}

		if (!bValid) {
			return NULL;
		}

		return &m_instance;
	}

private:

	inline bool checkLabel(const string& label) {
		if (label.length() > 2 && label[0] == 'b' && label[1] == '-') {
			return true;
		}
		if (label.length() == 1 && label[0] == 'i') {
			return true;
		}
		return false;
	}

};

#endif

