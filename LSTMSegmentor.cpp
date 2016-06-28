/*
 * Segmentor.cpp
 *
 *  Created on: Oct 23, 2015
 *      Author: mszhang
 */

#include "LSTMSegmentor.h"

#include "Argument_helper.h"

Segmentor::Segmentor() {
	// TODO Auto-generated constructor stub
	nullkey = "-null-";
	unknownkey = "-unknown-";
	paddingtag = "-padding-";
	seperateKey = "#";
}

Segmentor::~Segmentor() {
	// TODO Auto-generated destructor stub
}

// all linear features are extracted from positive examples
int Segmentor::createAlphabet(const vector<Instance>& vecInsts) {
	cout << "Creating Alphabet..." << endl;

	int numInstance = vecInsts.size();

	hash_map<string, int> char_stat;
	hash_map<string, int> bichar_stat;
	hash_map<string, int> action_stat;
	hash_map<string, int> feat_stat;
	hash_map<string, int> postag_stat;

	assert(numInstance > 0);

	static Metric segEval, posEval;
	static CStateItem state[m_classifier.MAX_SENTENCE_SIZE];
	static Feature feat;
	static CResult output;
	static CAction answer;
	static int actionNum;
	m_classifier.initAlphabet();
	segEval.reset();
	posEval.reset();
	int maxFreqChar = -1;
	int maxFreqWord = -1;

	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance &instance = vecInsts[numInstance];
		for (int idx = 0; idx < instance.postagsize(); idx++) {
			postag_stat[instance.postags[idx]];
			m_classifier.fe._tagConstraints.addWordPOSPair(instance.words[idx], instance.postags[idx]);
		}

		for (int idx = 0; idx < instance.charsize(); idx++) {
			char_stat[instance.chars[idx]]++;
		}

		for (int idx = 0; idx < instance.charsize() - 1; idx++) {
			bichar_stat[instance.chars[idx] + instance.chars[idx + 1]]++;
		}
		bichar_stat[instance.chars[instance.charsize() - 1] + m_classifier.fe.nullkey]++;
		bichar_stat[m_classifier.fe.nullkey + instance.chars[0]]++;
	}

	m_classifier.addToPostagAlphabet(postag_stat);

	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance &instance = vecInsts[numInstance];
		actionNum = 0;
		state[actionNum].initSentence(&instance.chars, &instance.candidateLabels);
		state[actionNum].clear();

		while (!state[actionNum].IsTerminated()) {
			state[actionNum].getGoldAction(instance, m_classifier.fe._postagAlphabet, answer);
			action_stat[answer.str()]++;

			m_classifier.extractFeature(state + actionNum, answer, feat);
			for (int idx = 0; idx < feat._strSparseFeat.size(); idx++) {
				feat_stat[feat._strSparseFeat[idx]]++;
			}
			state[actionNum].move(state + actionNum + 1, answer, m_classifier.fe._postagAlphabet);
			actionNum++;
		}

		if (actionNum - 1 != instance.charsize()) {
			std::cout << "action number is not correct, please check" << std::endl;
		}
		state[actionNum].getSegPosResults(output);

		instance.evaluate(output, segEval, posEval);

		if (!segEval.bIdentical() || !posEval.bIdentical()) {
			std::cout << "error state conversion!" << std::endl;
			std::cout << "output instance:" << std::endl;
			for (int tmpK = 0; tmpK < instance.words.size(); tmpK++) {
				std::cout << instance.words[tmpK] << "_" << instance.postags[tmpK] << " ";
			}
			std::cout << std::endl;

			std::cout << "predicated instance:" << std::endl;
			for (int tmpK = 0; tmpK < output.size(); tmpK++) {
				std::cout << output.words[tmpK] << "_" << output.postags[tmpK] << " ";
			}
			std::cout << std::endl;

			exit(0);
		}

		if ((numInstance + 1) % m_options.verboseIter == 0) {
			cout << numInstance + 1 << " ";
			if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
				cout << std::endl;
			cout.flush();
		}
		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}


	m_classifier.addToActionAlphabet(action_stat);
	m_classifier.addToCharAlphabet(char_stat, m_options.charEmbFineTune ? m_options.charCutOff : 0);
	m_classifier.addToBiCharAlphabet(bichar_stat, m_options.bicharEmbFineTune ? m_options.bicharCutOff : 0);
	m_classifier.addToFeatureAlphabet(feat_stat, m_options.featCutOff);

	cout << numInstance << " " << endl;
	cout << "Action num: " << m_classifier.fe._actionAlphabet.size() << endl;
	cout << "Pos num: " << m_classifier.fe._postagAlphabet.size() << endl;
	cout << "Total char num: " << char_stat.size() << endl;
	cout << "Total bichar num: " << bichar_stat.size() << endl;
	cout << "Total feat num: " << feat_stat.size() << endl;

	cout << "Remain char num: " << m_classifier.fe._charAlphabet.size() << endl;
	cout << "Remain bichar num: " << m_classifier.fe._bicharAlphabet.size() << endl;
	cout << "Remain feat num: " << m_classifier.fe._featAlphabet.size() << endl;

	//m_classifier.setFeatureCollectionState(false);

	return 0;
}

int Segmentor::addTestWordAlpha(const vector<Instance>& vecInsts) {
	cout << "Add test Alphabet..." << endl;

	hash_map<string, int> char_stat;
	hash_map<string, int> bichar_stat;
	int numInstance;

	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance &instance = vecInsts[numInstance];

		for (int idx = 0; idx < instance.charsize(); idx++) {
			char_stat[instance.chars[idx]]++;
		}
		for (int idx = 0; idx < instance.charsize() - 1; idx++) {
			bichar_stat[instance.chars[idx] + instance.chars[idx + 1]]++;
		}
		bichar_stat[instance.chars[instance.charsize() - 1] + m_classifier.fe.nullkey]++;
		bichar_stat[m_classifier.fe.nullkey + instance.chars[0]]++;

		if ((numInstance + 1) % m_options.verboseIter == 0) {
			cout << numInstance + 1 << " ";
			if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
				cout << std::endl;
			cout.flush();
		}
		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}

	m_classifier.addToCharAlphabet(char_stat, 0);
	m_classifier.addToBiCharAlphabet(bichar_stat, 0);

	cout << "Remain char num: " << m_classifier.fe._charAlphabet.size() << endl;
	cout << "Remain bichar num: " << m_classifier.fe._bicharAlphabet.size() << endl;

	return 0;
}

void Segmentor::getGoldActions(const vector<Instance>& vecInsts, vector<vector<CAction> >& vecActions) {
	vecActions.clear();

	static Metric segEval, posEval;
	static CStateItem state[m_classifier.MAX_SENTENCE_SIZE];
	static CResult output;
	static CAction answer;
	segEval.reset();
	posEval.reset();
	static int numInstance, actionNum;
	vecActions.resize(vecInsts.size());
	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance &instance = vecInsts[numInstance];

		actionNum = 0;
		state[actionNum].initSentence(&instance.chars, &instance.candidateLabels);
		state[actionNum].clear();

		while (!state[actionNum].IsTerminated()) {
			state[actionNum].getGoldAction(instance, m_classifier.fe._postagAlphabet, answer);
			vecActions[numInstance].push_back(answer);
			state[actionNum].move(state + actionNum + 1, answer, m_classifier.fe._postagAlphabet);
			actionNum++;
		}

		if (actionNum - 1 != instance.charsize()) {
			std::cout << "action number is not correct, please check" << std::endl;
		}
		state[actionNum].getSegPosResults(output);

		instance.evaluate(output, segEval, posEval);

		if (!segEval.bIdentical() || !posEval.bIdentical()) {
			std::cout << "error state conversion!" << std::endl;
			std::cout << "output instance:" << std::endl;
			for (int tmpK = 0; tmpK < instance.words.size(); tmpK++) {
				std::cout << instance.words[tmpK] << "_" << instance.postags[tmpK] << " ";
			}
			std::cout << std::endl;

			std::cout << "predicated instance:" << std::endl;
			for (int tmpK = 0; tmpK < output.size(); tmpK++) {
				std::cout << output.words[tmpK] << "_" << output.postags[tmpK] << " ";
			}
			std::cout << std::endl;

			exit(0);
		}

		if ((numInstance + 1) % m_options.verboseIter == 0) {
			cout << numInstance + 1 << " ";
			if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
				cout << std::endl;
			cout.flush();
		}
		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}
}

void Segmentor::train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile,
		const string& wordEmbFile, const string& charEmbFile, const string& bicharEmbFile, const string& lexiconFile) {
	if (optionFile != "")
		m_options.load(optionFile);

	m_options.showOptions();
	vector<Instance> trainInsts, devInsts, testInsts;
	m_pipe.readInstances(trainFile, trainInsts, m_classifier.MAX_SENTENCE_SIZE - 2, m_options.maxInstance);
	if (devFile != "")
		m_pipe.readInstances(devFile, devInsts, m_classifier.MAX_SENTENCE_SIZE - 2, m_options.maxInstance);
	if (testFile != "")
		m_pipe.readInstances(testFile, testInsts, m_classifier.MAX_SENTENCE_SIZE - 2, m_options.maxInstance);

	vector<vector<Instance> > otherInsts(m_options.testFiles.size());
	for (int idx = 0; idx < m_options.testFiles.size(); idx++) {
		m_pipe.readInstances(m_options.testFiles[idx], otherInsts[idx], m_classifier.MAX_SENTENCE_SIZE - 2, m_options.maxInstance);
	}

	createAlphabet(trainInsts);

	addTestWordAlpha(devInsts);
	addTestWordAlpha(testInsts);
	for (int idx = 0; idx < otherInsts.size(); idx++) {
		addTestWordAlpha(otherInsts[idx]);
	}

	NRMat<dtype> allwordEmb;
	if (wordEmbFile != "") {
		allWordAlphaEmb(wordEmbFile, allwordEmb);
	} else {
		std::cout << "must not be here, allword must be pretrained." << std::endl;
	}

	NRMat<dtype> charEmb;
	if (charEmbFile != "") {
		readEmbeddings(m_classifier.fe._charAlphabet, charEmbFile, charEmb);
	} else {
		charEmb.resize(m_classifier.fe._charAlphabet.size(), m_options.charEmbSize);
		charEmb.randu(2000);
	}

	cout << "char emb dim is " << charEmb.ncols() << std::endl;

	NRMat<dtype> bicharEmb;
	if (bicharEmbFile != "") {
		readEmbeddings(m_classifier.fe._bicharAlphabet, bicharEmbFile, bicharEmb);
	} else {
		bicharEmb.resize(m_classifier.fe._bicharAlphabet.size(), m_options.bicharEmbSize);
		bicharEmb.randu(3000);
	}

	cout << "bichar emb dim is " << bicharEmb.ncols() << std::endl;

	NRMat<dtype> actionEmb;
	actionEmb.resize(m_classifier.fe._actionAlphabet.size(), m_options.actionEmbSize);
	actionEmb.randu(4000);

	cout << "action emb dim is " << actionEmb.ncols() << std::endl;

	NRMat<dtype> lengthEmb;
	lengthEmb.resize(6, m_options.lengthEmbSize);
	lengthEmb.randu(5000);

	cout << "length emb dim is " << lengthEmb.ncols() << std::endl;

	NRMat<dtype> postagEmb;
	postagEmb.resize(m_classifier.fe._postagAlphabet.size(), m_options.postagEmbSize);
	postagEmb.randu(6000);

	cout << "postag emb dim is " << postagEmb.ncols() << std::endl;

	m_classifier.init(allwordEmb, m_options.wordNgram, m_options.wordHiddenSize, m_options.wordRNNHiddenSize,
			charEmb, bicharEmb, m_options.charcontext, m_options.charHiddenSize, m_options.charRNNHiddenSize,
			postagEmb, m_options.postagNgram, m_options.postagHiddenSize, m_options.postagRNNHiddenSize,
			actionEmb, m_options.actionNgram, m_options.actionHiddenSize, m_options.actionRNNHiddenSize,
			m_options.sepHiddenSize, m_options.appHiddenSize, m_options.delta);

	m_classifier.setDropValue(m_options.dropProb);


	vector<vector<CAction> > trainInstGoldactions;
	getGoldActions(trainInsts, trainInstGoldactions);
	double bestPostagFmeasure = 0;

	int inputSize = trainInsts.size();

	std::vector<int> indexes;
	for (int i = 0; i < inputSize; ++i)
		indexes.push_back(i);

	static Metric eval;
	static Metric segMetric_dev, segMetric_test;
	static Metric postagMetric_dev, postagMetric_test;

	int maxIter = m_options.maxIter * (inputSize / m_options.batchSize + 1);
	int oneIterMaxRound = (inputSize + m_options.batchSize - 1) / m_options.batchSize;
	std::cout << "maxIter = " << maxIter << std::endl;
	int devNum = devInsts.size(), testNum = testInsts.size();

	static vector<CResult> decodeInstResults;
	static CResult curDecodeInst;
	static bool bCurIterBetter;
	static vector<Instance > subInstances;
	static vector<vector<CAction> > subInstGoldActions;

	for (int iter = 0; iter < maxIter; ++iter) {
		std::cout << "##### Iteration " << iter << std::endl;
		srand(iter);
		random_shuffle(indexes.begin(), indexes.end());
		std::cout << "random: " << indexes[0] << ", " << indexes[indexes.size() - 1] << std::endl;
		bool bEvaluate = false;
		if (m_options.batchSize == 1) {
			eval.reset();
			bEvaluate = true;
			for (int idy = 0; idy < inputSize; idy++) {
				subInstances.clear();
				subInstGoldActions.clear();
				subInstances.push_back(trainInsts[indexes[idy]]);
				subInstGoldActions.push_back(trainInstGoldactions[indexes[idy]]);

				double cost = m_classifier.train(subInstances, subInstGoldActions);

				eval.overall_label_count += m_classifier._eval.overall_label_count;
				eval.correct_label_count += m_classifier._eval.correct_label_count;

				if ((idy + 1) % (m_options.verboseIter * 10) == 0) {
					std::cout << "current: " << idy + 1 << ", Cost = " << cost << ", Correct(%) = " << eval.getAccuracy() << std::endl;
				}
				m_classifier.updateParams(m_options.regParameter, m_options.adaAlpha, m_options.adaEps);
			}
			std::cout << "current: " << iter + 1 << ", Correct(%) = " << eval.getAccuracy() << std::endl;
		}
		else {
			if (iter == 0)
				eval.reset();
			subInstances.clear();
			subInstGoldActions.clear();
			for (int idy = 0; idy < m_options.batchSize; idy++) {
				subInstances.push_back(trainInsts[indexes[idy]]);
				subInstGoldActions.push_back(trainInstGoldactions[indexes[idy]]);
			}
			double cost = m_classifier.train(subInstances, subInstGoldActions);

			eval.overall_label_count += m_classifier._eval.overall_label_count;
			eval.correct_label_count += m_classifier._eval.correct_label_count;

			if ((iter + 1) % (m_options.verboseIter) == 0) {
				std::cout << "current: " << iter + 1 << ", Cost = " << cost << ", Correct(%) = " << eval.getAccuracy() << std::endl;
				eval.reset();
				bEvaluate = true;
			}

			m_classifier.updateParams(m_options.regParameter, m_options.adaAlpha, m_options.adaEps);
		}

		if (bEvaluate && devNum > 0) {
			bCurIterBetter = false;
			if (!m_options.outBest.empty())
				decodeInstResults.clear();
			segMetric_dev.reset();
			postagMetric_dev.reset();
			for (int idx = 0; idx < devInsts.size(); idx++) {
				predict(devInsts[idx], curDecodeInst);
				devInsts[idx].evaluate(curDecodeInst, segMetric_dev, postagMetric_dev);
				if (!m_options.outBest.empty()) {
					decodeInstResults.push_back(curDecodeInst);
				}
			}
			std::cout << "dev:" << std::endl << "Seg: ";
			segMetric_dev.print();
			std::cout << "Postag: ";
			postagMetric_dev.print();

			if (!m_options.outBest.empty() && postagMetric_dev.getAccuracy() > bestPostagFmeasure) {
				m_pipe.outputAllInstances(devFile + m_options.outBest, decodeInstResults);
				bCurIterBetter = true;
			}

			if (testNum > 0) {
				if (!m_options.outBest.empty())
					decodeInstResults.clear();
				segMetric_test.reset();
				postagMetric_test.reset();
				for (int idx = 0; idx < testInsts.size(); idx++) {
					predict(testInsts[idx], curDecodeInst);
					testInsts[idx].evaluate(curDecodeInst, segMetric_test, postagMetric_test);
					if (bCurIterBetter && !m_options.outBest.empty()) {
						decodeInstResults.push_back(curDecodeInst);
					}
				}
				std::cout << "test:" << std::endl << "Seg: ";
				segMetric_test.print();
				std::cout << "Postag: ";
				postagMetric_test.print();

				if (!m_options.outBest.empty() && bCurIterBetter) {
					m_pipe.outputAllInstances(testFile + m_options.outBest, decodeInstResults);
				}
			}

			for (int idx = 0; idx < otherInsts.size(); idx++) {
				std::cout << "processing " << m_options.testFiles[idx] << std::endl;
				if (!m_options.outBest.empty())
					decodeInstResults.clear();
				segMetric_test.reset();
				postagMetric_test.reset();
				for (int idy = 0; idy < otherInsts[idx].size(); idy++) {
					predict(otherInsts[idx][idy], curDecodeInst);
					otherInsts[idx][idy].evaluate(curDecodeInst, segMetric_test, postagMetric_test);
					if (bCurIterBetter && !m_options.outBest.empty()) {
						decodeInstResults.push_back(curDecodeInst);
					}
				}
				std::cout << "test:" << std::endl << "Seg: ";
				segMetric_test.print();
				std::cout << "Postag: ";
				postagMetric_test.print();

				if (!m_options.outBest.empty() && bCurIterBetter) {
					m_pipe.outputAllInstances(m_options.testFiles[idx] + m_options.outBest, decodeInstResults);
				}
			}

			if (m_options.saveIntermediate && postagMetric_dev.getAccuracy() > bestPostagFmeasure) {
				std::cout << "Exceeds best previous DIS of " << bestPostagFmeasure << ". Saving model file.." << std::endl;
				bestPostagFmeasure = postagMetric_dev.getAccuracy();
				writeModelFile(modelFile);
			}
		}
	}
}

void Segmentor::predict(const Instance& input, CResult& output) {
	m_classifier.decode(input, output);
}

void Segmentor::test(const string& testFile, const string& outputFile, const string& modelFile, const string& lexiconFile) {
	loadModelFile(modelFile);
	
	vector<Instance> testInsts;
	m_pipe.readInstances(testFile, testInsts, m_options.maxInstance);

	vector<CResult> testInstResults(testInsts.size());
	Metric segMetric_test, posMetric_test;
	segMetric_test.reset();
	posMetric_test.reset();
	for (int idx = 0; idx < testInsts.size(); idx++) {
		vector<string> result_labels;
		predict(testInsts[idx], testInstResults[idx]);
		testInsts[idx].evaluate(testInstResults[idx], segMetric_test, posMetric_test);
	}
	std::cout << "test:" << std::endl;
	segMetric_test.print();
	posMetric_test.print();

	std::ofstream os(outputFile.c_str());

	for (int idx = 0; idx < testInsts.size(); idx++) {
		for (int idy = 0; idy < testInstResults[idx].size(); idy++) {
			os << testInstResults[idx].words[idy] << "_" << testInstResults[idx].postags[idy] << " ";
		}
		os << std::endl;
	}
	os.close();
}

void Segmentor::loadModelFile(const string& inputModelFile) {

}

void Segmentor::writeModelFile(const string& outputModelFile) {

}

void Segmentor::readEmbeddings(Alphabet &alpha, const string& inFile, NRMat<dtype>& emb) {
	static ifstream inf;
	if (inf.is_open()) {
		inf.close();
		inf.clear();
	}
	inf.open(inFile.c_str());

	static string strLine, curWord;
	static int wordId;

	//find the first line, decide the wordDim;
	while (1) {
		if (!my_getline(inf, strLine)) {
			break;
		}
		if (!strLine.empty())
			break;
	}

	int unknownId = alpha.from_string(m_classifier.fe.unknownkey);

	static vector<string> vecInfo;
	split_bychar(strLine, vecInfo, ' ');
	int wordDim = vecInfo.size() - 1;

	std::cout << "embedding dim is " << wordDim << std::endl;

	emb.resize(alpha.size(), wordDim);
	emb = 0.0;
	curWord = normalize_to_lowerwithdigit(vecInfo[0]);
	wordId = alpha.from_string(curWord);
	hash_set<int> indexers;
	dtype sum[wordDim];
	int count = 0;
	bool bHasUnknown = false;
	if (wordId >= 0) {
		count++;
		if (unknownId == wordId)
			bHasUnknown = true;
		indexers.insert(wordId);
		for (int idx = 0; idx < wordDim; idx++) {
			dtype curValue = atof(vecInfo[idx + 1].c_str());
			sum[idx] = curValue;
			emb[wordId][idx] = curValue;
		}

	} else {
		for (int idx = 0; idx < wordDim; idx++) {
			sum[idx] = 0.0;
		}
	}

	while (1) {
		if (!my_getline(inf, strLine)) {
			break;
		}
		if (strLine.empty())
			continue;
		split_bychar(strLine, vecInfo, ' ');
		if (vecInfo.size() != wordDim + 1) {
			std::cout << "error embedding file" << std::endl;
		}
		curWord = normalize_to_lowerwithdigit(vecInfo[0]);
		wordId = alpha.from_string(curWord);
		if (wordId >= 0) {
			count++;
			if (unknownId == wordId)
				bHasUnknown = true;
			indexers.insert(wordId);

			for (int idx = 0; idx < wordDim; idx++) {
				dtype curValue = atof(vecInfo[idx + 1].c_str());
				sum[idx] += curValue;
				emb[wordId][idx] += curValue;
			}
		}

	}

	if (!bHasUnknown) {
		for (int idx = 0; idx < wordDim; idx++) {
			emb[unknownId][idx] = sum[idx] / count;
		}
		count++;
		std::cout << unknownkey << " not found, using averaged value to initialize." << std::endl;
	}

	int oovWords = 0;
	int totalWords = 0;
	for (int id = 0; id < alpha.size(); id++) {
		if (indexers.find(id) == indexers.end()) {
			oovWords++;
			for (int idx = 0; idx < wordDim; idx++) {
				emb[id][idx] = emb[unknownId][idx];
			}
		}
		totalWords++;
	}

	std::cout << "OOV num is " << oovWords << ", total num is " << alpha.size() << ", embedding oov ratio is " << oovWords * 1.0 / alpha.size()
			<< std::endl;

}

int Segmentor::allWordAlphaEmb(const string& inFile, NRMat<dtype>& emb) {
	cout << "All word  alphabet and emb creating..." << endl;

	hash_map<string, int> word_stat;

	static ifstream inf;
	if (inf.is_open()) {
		inf.close();
		inf.clear();
	}
	inf.open(inFile.c_str());

	static string strLine, curWord;
	static int wordId;
	static vector<string> vecInfo;
	vector<string> allLines;

	int wordDim = 0;
	while (1) {
		if (!my_getline(inf, strLine)) {
			break;
		}
		if (!strLine.empty()) {
			split_bychar(strLine, vecInfo, ' ');
			if (wordDim == 0) {
				wordDim = vecInfo.size() - 1;
				std::cout << "allword embedding dim is " << wordDim << std::endl;
			}
			curWord = normalize_to_lowerwithdigit(vecInfo[0]);
			word_stat[curWord]++;
			allLines.push_back(strLine);
		}
	}

	m_classifier.addToAllWordAlphabet(word_stat);
	cout << "Remain all word num: " << m_classifier.fe._allwordAlphabet.size() << endl;

	emb.resize(m_classifier.fe._allwordAlphabet.size(), wordDim);
	emb = 0.0;

	int unknownId = m_classifier.fe._allwordAlphabet.from_string(m_classifier.fe.unknownkey);
	dtype sum[wordDim];
	int count = 0;
	bool bHasUnknown = false;
	for (int idx = 0; idx < wordDim; idx++) {
		sum[idx] = 0.0;
	}

	for (int idx = 0; idx < allLines.size(); idx++) {
		split_bychar(allLines[idx], vecInfo, ' ');
		if (vecInfo.size() != wordDim + 1) {
			std::cout << "error embedding file" << std::endl;
		}
		curWord = normalize_to_lowerwithdigit(vecInfo[0]);
		wordId = m_classifier.fe._allwordAlphabet.from_string(curWord);
		if (wordId >= 0) {
			count++;
			if (unknownId == wordId)
				bHasUnknown = true;

			for (int idx = 0; idx < wordDim; idx++) {
				dtype curValue = atof(vecInfo[idx + 1].c_str());
				sum[idx] += curValue;
				emb[wordId][idx] += curValue;
			}
		}
		else {
			std::cout << "read all word embedding strange...." << std::endl;
		}

	}

	if (!bHasUnknown) {
		for (int idx = 0; idx < wordDim; idx++) {
			emb[unknownId][idx] = sum[idx] / count;
		}
		count++;
		std::cout << unknownkey << " not found, using averaged value to initialize." << std::endl;
	}

	return 0;
}

int main(int argc, char* argv[]) {
	std::string trainFile = "", devFile = "", testFile = "", modelFile = "";
	std::string wordEmbFile = "", charEmbFile = "", bicharEmbFile = "", optionFile = "", lexiconFile = "";
	std::string outputFile = "";
	bool bTrain = false;
	dsr::Argument_helper ah;

	ah.new_flag("l", "learn", "train or test", bTrain);
	ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training", trainFile);
	ah.new_named_string("dev", "devCorpus", "named_string", "development corpus to train a model, optional when training", devFile);
	ah.new_named_string("test", "testCorpus", "named_string",
			"testing corpus to train a model or input file to test a model, optional when training and must when testing", testFile);
	ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
	ah.new_named_string("word", "wordEmbFile", "named_string", "pretrained word embedding file to train a model, optional when training", wordEmbFile);
	ah.new_named_string("char", "charEmbFile", "named_string", "pretrained char embedding file to train a model, optional when training", charEmbFile);
	ah.new_named_string("bichar", "bicharEmbFile", "named_string", "pretrained bichar embedding file to train a model, optional when training", bicharEmbFile);
	ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
	ah.new_named_string("output", "outputFile", "named_string", "output file to test, must when testing", outputFile);
	 ah.new_named_string("lexicon", "lexiconFile", "named_string", "lexicon file, must file or leads to low performances", lexiconFile);

	ah.process(argc, argv);

	Segmentor segmentor;
	if (bTrain) {
		segmentor.train(trainFile, devFile, testFile, modelFile, optionFile, wordEmbFile, charEmbFile, bicharEmbFile, lexiconFile);
	} else {
		segmentor.test(testFile, outputFile, modelFile, lexiconFile);
	}

	//test(argv);
	//ah.write_values(std::cout);

}
