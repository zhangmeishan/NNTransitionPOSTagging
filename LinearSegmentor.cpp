/*
 * Segmentor.cpp
 *
 *  Created on: Oct 23, 2015
 *      Author: mszhang
 */

#include "LinearSegmentor.h"

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
	m_classifier.addToFeatureAlphabet(feat_stat, m_options.featCutOff);

	cout << numInstance << " " << endl;
	cout << "Action num: " << m_classifier.fe._actionAlphabet.size() << endl;
	cout << "Pos num: " << m_classifier.fe._postagAlphabet.size() << endl;
	cout << "Total feat num: " << feat_stat.size() << endl;

	cout << "Remain feat num: " << m_classifier.fe._featAlphabet.size() << endl;

	//m_classifier.setFeatureCollectionState(false);

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

void Segmentor::train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile, const string& lexiconFile) {
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

	m_classifier.init(m_options.delta);
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

int main(int argc, char* argv[]) {
	std::string trainFile = "", devFile = "", testFile = "", modelFile = "";
	std::string optionFile = "", lexiconFile = "";
	std::string outputFile = "";
	bool bTrain = false;
	dsr::Argument_helper ah;

	ah.new_flag("l", "learn", "train or test", bTrain);
	ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training", trainFile);
	ah.new_named_string("dev", "devCorpus", "named_string", "development corpus to train a model, optional when training", devFile);
	ah.new_named_string("test", "testCorpus", "named_string",
			"testing corpus to train a model or input file to test a model, optional when training and must when testing", testFile);
	ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
	ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
	ah.new_named_string("output", "outputFile", "named_string", "output file to test, must when testing", outputFile);
	 ah.new_named_string("lexicon", "lexiconFile", "named_string", "lexicon file, must file or leads to low performances", lexiconFile);

	ah.process(argc, argv);

	Segmentor segmentor;
	if (bTrain) {
		segmentor.train(trainFile, devFile, testFile, modelFile, optionFile, lexiconFile);
	} else {
		segmentor.test(testFile, outputFile, modelFile, lexiconFile);
	}

	//test(argv);
	//ah.write_values(std::cout);

}
