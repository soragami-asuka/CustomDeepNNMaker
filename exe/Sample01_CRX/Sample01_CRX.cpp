//=============================================
// クレジットカード認証のデータを用いた実装サンプル
// 参考URL：
// ・Dropout：ディープラーニングの火付け役、単純な方法で過学習を防ぐ
//	https://deepage.net/deep_learning/2016/10/17/deeplearning_dropout.html
//
// サンプルデータURL:
// https://archive.ics.uci.edu/ml/datasets/Credit+Approval
//  データ本体
//		Data Folder > crx.data
//  データフォーマットについて
//		Data Folder > crx.names
//=============================================


#include "stdafx.h"

#include <boost/tokenizer.hpp>
#include<boost/algorithm/string.hpp>
#include<boost/uuid/uuid_generators.hpp>


#include"Library/DataFormat/StringArray.h"
#include"Library/NeuralNetwork/LayerDLLManager.h"
#include"Library/NeuralNetwork/LayerDataManager.h"
#include"Library/Common/BatchDataNoListGenerator.h"
#include"Library/Layer/IOData/IODataLayer.h"
#include"Layer/Connect/ILayerConnectData.h"
#include"Layer/NeuralNetwork/INeuralNetwork.h"
#include"Utility/NeuralNetworkLayer.h"


#define USE_GPU	1
#define USE_HOST_MEMORY 1


using namespace Gravisbell;

/** サンプルデータの読み込み */
DataFormat::IDataFormatBase* LoadSampleData(const std::wstring& formatFilePath, const std::wstring& dataFilePath);

/** ニューラルネットワーククラスを作成する */
Layer::Connect::ILayerConnectData* CreateNeuralNetwork(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	const IODataStruct& inputDataStruct, const IODataStruct& outputDataStruct);

/** ニューラルネットワークの学習 */
Gravisbell::ErrorCode LearnNeuralNetwork(
	Layer::NeuralNetwork::INeuralNetwork* pNeuralNetwork,
	Layer::IOData::IIODataLayer* pInputLayer,
	Layer::IOData::IIODataLayer* pTeachLayer,
	const U32 BATCH_SIZE,
	const U32 LEARN_TIMES);

/** サンプルデータとの誤差計測 */
Gravisbell::ErrorCode CalculateSampleError(
	Layer::NeuralNetwork::INeuralNetwork* pNeuralNetwork,
	Layer::IOData::IIODataLayer* pInputLayer,
	Layer::IOData::IIODataLayer* pTeachLayer);

/** ニューラルネットワークの学習とサンプル実行を同時実行 */
Gravisbell::ErrorCode LearnWithCalculateSampleError(
	Layer::NeuralNetwork::INeuralNetwork* pNeuralNetworkLearn,
	Layer::NeuralNetwork::INeuralNetwork* pNeuralNetworkSample,
	Layer::IOData::IIODataLayer* pTeachInputLayer,
	Layer::IOData::IIODataLayer* pTeachTeachLayer,
	Layer::IOData::IIODataLayer* pSampleInputLayer,
	Layer::IOData::IIODataLayer* pSampleTeachLayer,
	const U32 BATCH_SIZE,
	const U32 LEARN_TIMES);



int _tmain(int argc, _TCHAR* argv[])
{
#ifdef _DEBUG
	::_CrtSetDbgFlag(_CRTDBG_LEAK_CHECK_DF | _CRTDBG_ALLOC_MEM_DF);
#endif


	srand(12345);

	// サンプルデータの読み込み
#ifndef _WIN64
	Gravisbell::DataFormat::IDataFormatBase* pSampleData = ::LoadSampleData(L"../SampleData/CRX/DataFormat.xml", L"../SampleData/CRX/crx.csv");
#else
	Gravisbell::DataFormat::IDataFormatBase* pSampleData = ::LoadSampleData(L"../../SampleData/CRX/DataFormat.xml", L"../../SampleData/CRX/crx.csv");
#endif
	printf("入力信号：%d\n", pSampleData->GetDataStruct(L"input").GetDataCount());
	printf("出力信号：%d\n", pSampleData->GetDataStruct(L"output").GetDataCount());

	// レイヤーDLL管理クラスを作成
#if USE_GPU
	Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager = Gravisbell::Utility::NeuralNetworkLayer::CreateLayerDLLManagerGPU(L"./");
#else
	Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager = Gravisbell::Utility::NeuralNetworkLayer::CreateLayerDLLManagerCPU(L"./");
#endif
	if(pLayerDLLManager == NULL)
	{
		delete pSampleData;
		return -1;
	}

	// レイヤーデータ管理クラスを作成
	Gravisbell::Layer::NeuralNetwork::ILayerDataManager* pLayerDataManager = Gravisbell::Layer::NeuralNetwork::CreateLayerDataManager();
	if(pLayerDataManager == NULL)
	{
		delete pLayerDLLManager;
		delete pSampleData;
		return -1;
	}


	// 入出力信号レイヤーを作成
#if USE_GPU
#if USE_HOST_MEMORY
	Layer::IOData::IIODataLayer* pTeachInputLayer  = Layer::IOData::CreateIODataLayerGPU_host( pSampleData->GetDataStruct(L"input") );	// 入力信号(教師信号)
	Layer::IOData::IIODataLayer* pTeachOutputLayer = Layer::IOData::CreateIODataLayerGPU_host( pSampleData->GetDataStruct(L"output") );	// 出力信号(教師信号)
	Layer::IOData::IIODataLayer* pSampleInputLayer  = Layer::IOData::CreateIODataLayerGPU_host( pSampleData->GetDataStruct(L"input") );	// 入力信号(確認用サンプルデータ)
	Layer::IOData::IIODataLayer* pSampleOutputLayer = Layer::IOData::CreateIODataLayerGPU_host( pSampleData->GetDataStruct(L"output") );	// 出力信号(確認用サンプルデータ)
#else
	Layer::IOData::IIODataLayer* pTeachInputLayer  = Layer::IOData::CreateIODataLayerGPU_device( pSampleData->GetDataStruct(L"input") );	// 入力信号(教師信号)
	Layer::IOData::IIODataLayer* pTeachOutputLayer = Layer::IOData::CreateIODataLayerGPU_device( pSampleData->GetDataStruct(L"output") );	// 出力信号(教師信号)
	Layer::IOData::IIODataLayer* pSampleInputLayer  = Layer::IOData::CreateIODataLayerGPU_device( pSampleData->GetDataStruct(L"input") );	// 入力信号(確認用サンプルデータ)
	Layer::IOData::IIODataLayer* pSampleOutputLayer = Layer::IOData::CreateIODataLayerGPU_device( pSampleData->GetDataStruct(L"output") );	// 出力信号(確認用サンプルデータ)
#endif
#else
	Layer::IOData::IIODataLayer* pTeachInputLayer  = Layer::IOData::CreateIODataLayerCPU( pSampleData->GetDataStruct(L"input") );	// 入力信号(教師信号)
	Layer::IOData::IIODataLayer* pTeachOutputLayer = Layer::IOData::CreateIODataLayerCPU( pSampleData->GetDataStruct(L"output") );	// 出力信号(教師信号)
	Layer::IOData::IIODataLayer* pSampleInputLayer  = Layer::IOData::CreateIODataLayerCPU( pSampleData->GetDataStruct(L"input") );	// 入力信号(確認用サンプルデータ)
	Layer::IOData::IIODataLayer* pSampleOutputLayer = Layer::IOData::CreateIODataLayerCPU( pSampleData->GetDataStruct(L"output") );	// 出力信号(確認用サンプルデータ)
#endif
	const F32 USE_TEACH_RATE = 0.8f;
	for(U32 dataNum=0; dataNum<(U32)(pSampleData->GetDataCount()*USE_TEACH_RATE); dataNum++)
	{
		pTeachInputLayer->AddData(pSampleData->GetDataByNum(dataNum, L"input"));
		pTeachOutputLayer->AddData(pSampleData->GetDataByNum(dataNum, L"output"));
	}
	for(U32 dataNum=(U32)(pSampleData->GetDataCount()*USE_TEACH_RATE); dataNum<pSampleData->GetDataCount(); dataNum++)
	{
		pSampleInputLayer->AddData(pSampleData->GetDataByNum(dataNum, L"input"));
		pSampleOutputLayer->AddData(pSampleData->GetDataByNum(dataNum, L"output"));
	}
	printf("訓練データ  ：%d\n", pTeachInputLayer->GetDataCount());
	printf("テストデータ：%d\n", pSampleInputLayer->GetDataCount());

	// ニューラルネットワーククラスを作成
	Layer::Connect::ILayerConnectData* pNeuralNetworkData = ::CreateNeuralNetwork(*pLayerDLLManager, *pLayerDataManager, pTeachInputLayer->GetOutputDataStruct(), pTeachOutputLayer->GetInputDataStruct());
	if(pNeuralNetworkData == NULL)
	{
		delete pTeachInputLayer;
		delete pTeachOutputLayer;
		delete pSampleInputLayer;
		delete pSampleOutputLayer;
		delete pLayerDataManager;
		delete pLayerDLLManager;
		delete pSampleData;
		return -1;
	}
	IODataStruct outputDataStruct = pNeuralNetworkData->GetOutputDataStruct(&pTeachInputLayer->GetOutputDataStruct(), 1);
	if(outputDataStruct != pTeachOutputLayer->GetInputDataStruct())
	{
		delete pTeachInputLayer;
		delete pTeachOutputLayer;
		delete pSampleInputLayer;
		delete pSampleOutputLayer;
		delete pLayerDataManager;
		delete pLayerDLLManager;
		delete pSampleData;
		return -1;
	}



	// 開始時刻を計測
	clock_t startTime = clock();

	// 学習,サンプル実行別実行
#if 0
	{
		// ニューラルネットワークを作成
		Layer::NeuralNetwork::INeuralNetwork* pNeuralNetwork = NULL;
		{
			Layer::NeuralNetwork::INNLayer* pLayer = pNeuralNetworkData->CreateLayer(boost::uuids::random_generator()().data);
			pNeuralNetwork = dynamic_cast<Layer::NeuralNetwork::INeuralNetwork*>(pLayer);
			if(pNeuralNetwork == NULL)
				delete pLayer;
		}
		if(pNeuralNetwork == NULL)
		{
			delete pTeachInputLayer;
			delete pTeachOutputLayer;
			delete pSampleInputLayer;
			delete pSampleOutputLayer;
			delete pLayerDataManager;
			delete pLayerDLLManager;
			delete pSampleData;
			return -1;
		}

		// 学習
		if(::LearnNeuralNetwork(pNeuralNetwork, pTeachInputLayer, pTeachOutputLayer, 1, 8000) != ErrorCode::ERROR_CODE_NONE)
		{
			delete pTeachInputLayer;
			delete pTeachOutputLayer;
			delete pSampleInputLayer;
			delete pSampleOutputLayer;
			delete pLayerDataManager;
			delete pLayerDLLManager;
			delete pSampleData;
			return -1;
		}

		// サンプルとの誤差計算
		if(::CalculateSampleError(pNeuralNetwork, pSampleInputLayer, pSampleOutputLayer))
		{
			delete pTeachInputLayer;
			delete pTeachOutputLayer;
			delete pSampleInputLayer;
			delete pSampleOutputLayer;
			delete pLayerDataManager;
			delete pLayerDLLManager;
			delete pSampleData;
			return -1;
		}

		// メモリ開放
		delete pNeuralNetwork;
	}
#else
	{
		// ニューラルネットワークを作成
		Layer::NeuralNetwork::INeuralNetwork* pNeuralNetworkLearn = NULL;
		{
			Layer::ILayerBase* pLayer = pNeuralNetworkData->CreateLayer(boost::uuids::random_generator()().data, &pTeachInputLayer->GetInputDataStruct(), 1);
			pNeuralNetworkLearn = dynamic_cast<Layer::NeuralNetwork::INeuralNetwork*>(pLayer);
			if(pNeuralNetworkLearn == NULL)
				delete pLayer;
		}
		if(pNeuralNetworkLearn == NULL)
		{
			delete pTeachInputLayer;
			delete pTeachOutputLayer;
			delete pSampleInputLayer;
			delete pSampleOutputLayer;
			delete pLayerDataManager;
			delete pLayerDLLManager;
			delete pSampleData;
			return -1;
		}
		
		Layer::NeuralNetwork::INeuralNetwork* pNeuralNetworkSample = NULL;
		{
			Layer::ILayerBase* pLayer = pNeuralNetworkData->CreateLayer(boost::uuids::random_generator()().data, &pSampleInputLayer->GetInputDataStruct(), 1);
			pNeuralNetworkSample = dynamic_cast<Layer::NeuralNetwork::INeuralNetwork*>(pLayer);
			if(pNeuralNetworkSample == NULL)
				delete pLayer;
		}
		if(pNeuralNetworkSample == NULL)
		{
			delete pNeuralNetworkLearn;
			delete pTeachInputLayer;
			delete pTeachOutputLayer;
			delete pSampleInputLayer;
			delete pSampleOutputLayer;
			delete pLayerDataManager;
			delete pLayerDLLManager;
			delete pSampleData;
			return -1;
		}

		// 学習
		if(::LearnWithCalculateSampleError(pNeuralNetworkLearn, pNeuralNetworkSample, pTeachInputLayer, pTeachOutputLayer, pSampleInputLayer, pSampleOutputLayer, 32, 100) != ErrorCode::ERROR_CODE_NONE)
		{
			delete pNeuralNetworkSample;
			delete pNeuralNetworkLearn;
			delete pTeachInputLayer;
			delete pTeachOutputLayer;
			delete pSampleInputLayer;
			delete pSampleOutputLayer;
			delete pLayerDataManager;
			delete pLayerDLLManager;
			delete pSampleData;
			return -1;
		}


		// メモリ開放
		delete pNeuralNetworkSample;
		delete pNeuralNetworkLearn;
	}
#endif

	// 終了時刻を計測
	clock_t endTime = clock();

	// 経過時間を表示
	printf("処理時間(ms) = %d\n", endTime - startTime);

	// メモリ開放
	delete pTeachInputLayer;
	delete pTeachOutputLayer;
	delete pSampleInputLayer;
	delete pSampleOutputLayer;
	delete pLayerDataManager;
	delete pLayerDLLManager;
	delete pSampleData;

	printf("Press any key to continue");
	getc(stdin);

	return 0;
}


/** サンプルデータの読み込み */
Gravisbell::DataFormat::IDataFormatBase* LoadSampleData(const std::wstring& formatFilePath, const std::wstring& dataFilePath)
{
	// フォーマットを読み込み
	auto pDataFormat = Gravisbell::DataFormat::StringArray::CreateDataFormatFromXML(formatFilePath.c_str());
	if(pDataFormat == NULL)
		return NULL;

	// CSVファイルを読み込んでフォーマットに追加
	{
		// ファイルオープン
		FILE* fp = _wfopen(dataFilePath.c_str(), L"r");
		if(fp == NULL)
		{
			delete pDataFormat;
			return NULL;
		}

		wchar_t szBuf[1024];
		while(fgetws(szBuf, sizeof(szBuf)/sizeof(wchar_t)-1, fp))
		{
			size_t len = wcslen(szBuf);
			if(szBuf[len-1] == '\n')
				szBuf[len-1] = NULL;

			// ","(カンマ)区切りで分離
			std::vector<std::wstring> lpBuf;
			boost::split(lpBuf, szBuf, boost::is_any_of(L","));

			std::vector<const wchar_t*> lpBufPointer;
			for(auto& buf : lpBuf)
				lpBufPointer.push_back(buf.c_str());


			pDataFormat->AddDataByStringArray(&lpBufPointer[0]);
		}

		// ファイルクローズ
		fclose(fp);
	}
	// 正規化
	pDataFormat->Normalize();

	return pDataFormat;
}

/** ニューラルネットワーククラスを作成する */
Layer::Connect::ILayerConnectData* CreateNeuralNetwork(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
{
	using namespace Gravisbell::Utility::NeuralNetworkLayer;

	Layer::Connect::ILayerConnectData* pNeuralNetwork = Gravisbell::Utility::NeuralNetworkLayer::CreateNeuralNetwork(layerDLLManager, layerDataManager);

	if(pNeuralNetwork)
	{
		// 入力信号を直前レイヤーに設定
		Gravisbell::IODataStruct inputDataStruct = i_inputDataStruct;
		Gravisbell::GUID lastLayerGUID = pNeuralNetwork->GetInputGUID();

		// 1層目
		AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateFullyConnectLayer(layerDLLManager, layerDataManager, inputDataStruct.GetDataCount(), 256));
		AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, L"ReLU"));
		inputDataStruct = pNeuralNetwork->GetOutputDataStruct(lastLayerGUID, &i_inputDataStruct, 1);
		AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateDropoutLayer(layerDLLManager, layerDataManager, 0.2f));

		// 2層目
		AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateFullyConnectLayer(layerDLLManager, layerDataManager, inputDataStruct.GetDataCount(), 256));
		AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, L"ReLU"));
		inputDataStruct = pNeuralNetwork->GetOutputDataStruct(lastLayerGUID, &i_inputDataStruct, 1);
		AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateDropoutLayer(layerDLLManager, layerDataManager, 0.5f));

		// 3層目(出力層)
		AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateFullyConnectLayer(layerDLLManager, layerDataManager, inputDataStruct.GetDataCount(), i_outputDataStruct.GetDataCount()));
		AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, L"softmax_ALL_crossEntropy"));

		// 出力レイヤー設定
		pNeuralNetwork->SetOutputLayerGUID(lastLayerGUID);
	}

	pNeuralNetwork->ChangeOptimizer(L"Adam");

	return pNeuralNetwork;
}


/** ニューラルネットワークの学習 */
Gravisbell::ErrorCode LearnNeuralNetwork(
	Layer::NeuralNetwork::INeuralNetwork* pNeuralNetwork,
	Layer::IOData::IIODataLayer* pInputLayer,
	Layer::IOData::IIODataLayer* pTeachLayer,
	const U32 BATCH_SIZE,
	const U32 LEARN_TIMES)
{
	Gravisbell::ErrorCode err;

	// 学習係数を設定
	pNeuralNetwork->SetLearnSettingData(L"LearnCoeff", 0.1f);

	// 事前処理を実行
	err = pNeuralNetwork->PreProcessLearn(BATCH_SIZE);
	if(err != ErrorCode::ERROR_CODE_NONE)
		return err;
	err = pInputLayer->PreProcessLearn(BATCH_SIZE);
	if(err != ErrorCode::ERROR_CODE_NONE)
		return err;
	err = pTeachLayer->PreProcessLearn(BATCH_SIZE);
	if(err != ErrorCode::ERROR_CODE_NONE)
		return err;


	// バッチNo生成クラスを作成
	Gravisbell::Common::IBatchDataNoListGenerator* pBatchDataNoListGenerator = Gravisbell::Common::CreateBatchDataNoListGenerator();
	err = pBatchDataNoListGenerator->PreProcess(pInputLayer->GetDataCount(), BATCH_SIZE);
	if(err != ErrorCode::ERROR_CODE_NONE)
	{
		delete pBatchDataNoListGenerator;
		return err;
	}

	// ダミーの学習設定を作成
	Gravisbell::SettingData::Standard::IData* pLearnSetting = Gravisbell::Layer::IOData::CreateLearningSetting();
	if(pLearnSetting == NULL)
	{
		delete pBatchDataNoListGenerator;
		return err;
	}

	// 学習を実行
	for(U32 learnTime=0; learnTime<LEARN_TIMES; learnTime++)
	{
//		printf("%4d回 ", learnTime);
		printf("%4d,",learnTime);

		// 学習ループ先頭処理
		pBatchDataNoListGenerator->PreProcessLearnLoop();
		pInputLayer->PreProcessLearnLoop(*pLearnSetting);
		pTeachLayer->PreProcessLearnLoop(*pLearnSetting);
		pNeuralNetwork->PreProcessLearnLoop(*pLearnSetting);

		// バッチ単位で処理
		for(U32 batchNum=0; batchNum<pBatchDataNoListGenerator->GetBatchDataNoListCount(); batchNum++)
		{
			// データ切り替え
			pInputLayer->SetBatchDataNoList(pBatchDataNoListGenerator->GetBatchDataNoListByNum(batchNum));
			pTeachLayer->SetBatchDataNoList(pBatchDataNoListGenerator->GetBatchDataNoListByNum(batchNum));

			// 演算
			pNeuralNetwork->Calculate(pInputLayer->GetOutputBuffer());

			// 誤差計算
			// 教師信号との誤差計算
			pTeachLayer->CalculateLearnError(pNeuralNetwork->GetOutputBuffer());

			// 学習
			pNeuralNetwork->Training(NULL, pTeachLayer->GetDInputBuffer());
		}

		// 誤差表示
		{
			F32 errorMax, errorAve, errorAve2, errorCrossEntoropy;
			pTeachLayer->GetCalculateErrorValue(errorMax, errorAve, errorAve2, errorCrossEntoropy);
//			printf("min=%.3f, max=%.3f, ave=%.3f, ave2=%.3f\n", errorMin, errorMax, errorAve, errorAve2);
			printf("%.3f,%.3f,%.3f,%.3f\n", errorMax, errorAve, errorAve2, errorCrossEntoropy); 
		}
	}

	// メモリ開放
	delete pLearnSetting;
	delete pBatchDataNoListGenerator;

	return ErrorCode::ERROR_CODE_NONE;
}

/** サンプルデータとの誤差計測 */
Gravisbell::ErrorCode CalculateSampleError(
	Layer::NeuralNetwork::INeuralNetwork* pNeuralNetwork,
	Layer::IOData::IIODataLayer* pInputLayer,
	Layer::IOData::IIODataLayer* pTeachLayer)
{
	Gravisbell::ErrorCode err;

	// 事前処理を実行
	err = pNeuralNetwork->PreProcessCalculate(1);
	if(err != ErrorCode::ERROR_CODE_NONE)
		return err;
	err = pInputLayer->PreProcessCalculate(1);
	if(err != ErrorCode::ERROR_CODE_NONE)
		return err;
	err = pTeachLayer->PreProcessCalculate(1);
	if(err != ErrorCode::ERROR_CODE_NONE)
		return err;

	// 演算ループ先頭処理
	pInputLayer->PreProcessCalculateLoop();
	pTeachLayer->PreProcessCalculateLoop();
	pNeuralNetwork->PreProcessCalculateLoop();

	// バッチ単位で処理
	for(U32 dataNum=0; dataNum<pInputLayer->GetDataCount(); dataNum++)
	{
		// データ切り替え
		pInputLayer->SetBatchDataNoList(&dataNum);
		pTeachLayer->SetBatchDataNoList(&dataNum);

		// 演算
		pNeuralNetwork->Calculate(pInputLayer->GetOutputBuffer());

		// 誤差計算
		pTeachLayer->CalculateLearnError(pNeuralNetwork->GetOutputBuffer());
	}

	// 誤差表示
	printf("\nサンプル誤差\n");
	{
		F32 errorMax, errorAve, errorAve2, errorCrossEntoropy;
		pTeachLayer->GetCalculateErrorValue(errorMax, errorAve, errorAve2, errorCrossEntoropy);
		printf("max=%.3f, ave=%.3f, ave2=%.3f\n", errorMax, errorAve, errorAve2);
	}

	return ErrorCode::ERROR_CODE_NONE;
}



/** ニューラルネットワークの学習とサンプル実行を同時実行 */
Gravisbell::ErrorCode LearnWithCalculateSampleError(
	Layer::NeuralNetwork::INeuralNetwork* pNeuralNetworkLearn,
	Layer::NeuralNetwork::INeuralNetwork* pNeuralNetworkSample,
	Layer::IOData::IIODataLayer* pTeachInputLayer,
	Layer::IOData::IIODataLayer* pTeachOutputLayer,
	Layer::IOData::IIODataLayer* pSampleInputLayer,
	Layer::IOData::IIODataLayer* pSampleOutputLayer,
	const U32 BATCH_SIZE,
	const U32 LEARN_TIMES)
{
	Gravisbell::ErrorCode err;

	// 学習係数を設定
	pNeuralNetworkLearn->SetLearnSettingData(L"LearnCoeff", 0.01f);

	// 事前処理を実行
	err = pNeuralNetworkLearn->PreProcessLearn(BATCH_SIZE);
	if(err != ErrorCode::ERROR_CODE_NONE)
		return err;
	err = pTeachInputLayer->PreProcessLearn(BATCH_SIZE);
	if(err != ErrorCode::ERROR_CODE_NONE)
		return err;
	err = pTeachOutputLayer->PreProcessLearn(BATCH_SIZE);
	if(err != ErrorCode::ERROR_CODE_NONE)
		return err;

	err = pNeuralNetworkSample->PreProcessCalculate(1);
	if(err != ErrorCode::ERROR_CODE_NONE)
		return err;
	err = pSampleInputLayer->PreProcessCalculate(1);
	if(err != ErrorCode::ERROR_CODE_NONE)
		return err;
	err = pSampleOutputLayer->PreProcessCalculate(1);
	if(err != ErrorCode::ERROR_CODE_NONE)
		return err;


	// バッチNo生成クラスを作成
	Gravisbell::Common::IBatchDataNoListGenerator* pBatchDataNoListGenerator = Gravisbell::Common::CreateBatchDataNoListGenerator();
	err = pBatchDataNoListGenerator->PreProcess(pTeachInputLayer->GetDataCount(), BATCH_SIZE);
	if(err != ErrorCode::ERROR_CODE_NONE)
	{
		delete pBatchDataNoListGenerator;
		return err;
	}

	// ダミーの学習設定を作成
	Gravisbell::SettingData::Standard::IData* pLearnSetting = Gravisbell::Layer::IOData::CreateLearningSetting();
	if(pLearnSetting == NULL)
	{
		delete pBatchDataNoListGenerator;
		return err;
	}

	std::vector<F32> lpOutputBuffer(pTeachOutputLayer->GetBufferCount() * BATCH_SIZE);
	std::vector<F32> lpTeachBuffer(pTeachOutputLayer->GetBufferCount() * BATCH_SIZE);

	// 学習を実行
	for(U32 learnTime=0; learnTime<LEARN_TIMES; learnTime++)
	{
//		printf("%5d回 ", learnTime);
		printf("%5d,", learnTime);

		U32 correctCount_learn  = 0;	// 正解数
		U32 correctCount_sample = 0;	// 正解数

		// 学習
		{
			// 学習ループ先頭処理
//			pBatchDataNoListGenerator->PreProcessLearnLoop();
			pTeachInputLayer->PreProcessLearnLoop(*pLearnSetting);
			pTeachOutputLayer->PreProcessLearnLoop(*pLearnSetting);
			pNeuralNetworkLearn->PreProcessLearnLoop(*pLearnSetting);

			// 学習処理
			// バッチ単位で処理
			for(U32 batchNum=0; batchNum<pBatchDataNoListGenerator->GetBatchDataNoListCount(); batchNum++)
			{
				// データ切り替え
				pTeachInputLayer->SetBatchDataNoList(pBatchDataNoListGenerator->GetBatchDataNoListByNum(batchNum));
				pTeachOutputLayer->SetBatchDataNoList(pBatchDataNoListGenerator->GetBatchDataNoListByNum(batchNum));

				// 演算
				pNeuralNetworkLearn->Calculate(pTeachInputLayer->GetOutputBuffer());

				// 誤差計算
				// 教師信号との誤差計算
				pTeachOutputLayer->CalculateLearnError(pNeuralNetworkLearn->GetOutputBuffer());

				// 学習
				pNeuralNetworkLearn->Training(NULL, pTeachOutputLayer->GetDInputBuffer());


				// 正解率を算出する
				pTeachOutputLayer->GetOutputBuffer(&lpTeachBuffer[0]);
				pNeuralNetworkLearn->GetOutputBuffer(&lpOutputBuffer[0]);
				for(U32 batchDataNum=0; batchDataNum<pTeachOutputLayer->GetBatchSize(); batchDataNum++)
				{
					// 正解の番号を取得
					U32 correctNo = 0;
					{
						F32 curValue = 0.0f;
						for(U32 i=0; i<pTeachOutputLayer->GetBufferCount(); i++)
						{
							U32 bufferPos = batchDataNum * pTeachOutputLayer->GetBufferCount() + i;

							if(lpTeachBuffer[bufferPos] > curValue)
							{
								correctNo = i;
								curValue = lpTeachBuffer[bufferPos];
							}
						}
					}
					// 出力された番号を取得
					U32 outputNo = 0;
					{
						F32 curValue = 0.0f;
						for(U32 i=0; i<pTeachOutputLayer->GetBufferCount(); i++)
						{
							U32 bufferPos = batchDataNum * pTeachOutputLayer->GetBufferCount() + i;

							if(lpOutputBuffer[bufferPos] > curValue)
							{
								outputNo = i;
								curValue = lpOutputBuffer[bufferPos];
							}
						}
					}

					if(correctNo == outputNo)
					{
						correctCount_learn++;
					}
				}
			}
		}


		// サンプル実行
		{		
			// サンプル実行先頭処理
			pSampleInputLayer->PreProcessCalculateLoop();
			pSampleOutputLayer->PreProcessCalculateLoop();
			pNeuralNetworkSample->PreProcessCalculateLoop();

			// バッチ単位で処理
			for(U32 dataNum=0; dataNum<pSampleInputLayer->GetDataCount(); dataNum++)
			{
				// データ切り替え
				pSampleInputLayer->SetBatchDataNoList(&dataNum);
				pSampleOutputLayer->SetBatchDataNoList(&dataNum);

				// 演算
				pNeuralNetworkSample->Calculate(pSampleInputLayer->GetOutputBuffer());

				// 誤差計算
				pSampleOutputLayer->CalculateLearnError(pNeuralNetworkSample->GetOutputBuffer());


				// 正解の番号を取得
				pSampleOutputLayer->GetOutputBuffer(&lpTeachBuffer[0]);
				pNeuralNetworkSample->GetOutputBuffer(&lpOutputBuffer[0]);
				{
					U32 correctNo = 0;
					{
						F32 curValue = 0.0f;
						for(U32 i=0; i<pSampleOutputLayer->GetBufferCount(); i++)
						{
							if(lpTeachBuffer[i] > curValue)
							{
								correctNo = i;
								curValue = lpTeachBuffer[i];
							}
						}
					}
					// 出力された番号を取得
					U32 outputNo = 0;
					{
						F32 curValue = 0.0f;
						for(U32 i=0; i<pSampleOutputLayer->GetBufferCount(); i++)
						{
							if(lpOutputBuffer[i] > curValue)
							{
								outputNo = i;
								curValue = lpOutputBuffer[i];
							}
						}
					}

					if(correctNo == outputNo)
					{
						correctCount_sample++;
					}
				}
			}
		}

		// 誤差表示
		{
			F32 errorMax, errorAve, errorAve2, errorCrossEntoropy;
			pTeachOutputLayer->GetCalculateErrorValue(errorMax, errorAve, errorAve2, errorCrossEntoropy);
//			printf("学習：max=%.3f, ave=%.3f, ave2=%.3f, entropy=%.3f", errorMax, errorAve, errorAve2, errorCrossEntoropy);
			printf("%.3f,%.3f,%.3f,%.3f,",  errorMax, errorAve2, errorCrossEntoropy, (F32)correctCount_learn / (pBatchDataNoListGenerator->GetBatchDataNoListCount() * BATCH_SIZE)); 
		}
//		printf(" : ");
		{
			F32 errorMax, errorAve, errorAve2, errorCrossEntoropy;
			pSampleOutputLayer->GetCalculateErrorValue(errorMax, errorAve, errorAve2, errorCrossEntoropy);
//			printf("実行：max=%.3f, ave=%.3f, ave2=%.3f, entropy=%.3f", errorMax, errorAve, errorAve2, errorCrossEntoropy);
			printf("%.3f,%.3f,%.3f,%.3f", errorMax, errorAve2, errorCrossEntoropy, (F32)correctCount_sample / pSampleInputLayer->GetDataCount()); 
		}
		printf("\n");
	}

	// メモリ開放
	delete pLearnSetting;
	delete pBatchDataNoListGenerator;

	return ErrorCode::ERROR_CODE_NONE;
}