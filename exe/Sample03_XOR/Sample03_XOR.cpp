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


#include"Library/DataFormat/DataFormatStringArray/DataFormat.h"
#include"Library/NeuralNetwork/LayerDLLManager/LayerDLLManager.h"
#include"Library/Common/BatchDataNoListGenerator/BatchDataNoListGenerator.h"
#include"Layer/IOData/IODataLayer/IODataLayer.h"
#include"Layer/NeuralNetwork/INNLayerData.h"
#include"Layer/NeuralNetwork/INNLayerConnectData.h"
#include"Layer/NeuralNetwork/INeuralNetwork.h"

using namespace Gravisbell;

/** サンプルデータの読み込み */
DataFormat::IDataFormatBase* LoadSampleData(const std::wstring& formatFilePath, const std::wstring& dataFilePath);
/** レイヤーDLL管理クラスの作成 */
Layer::NeuralNetwork::ILayerDLLManager* CreateLayerDLLManager(void);

/** レイヤーデータを作成 */
Layer::NeuralNetwork::INNLayerData* CreateLayerData(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, U32 neuronCount, U32 inputDataCount, const std::wstring activationType);
Layer::NeuralNetwork::INNLayerData* CreateHiddenLayerData(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, U32 neuronCount, U32 inputDataCount);
Layer::NeuralNetwork::INNLayerData* CreateOutputLayerData(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, U32 neuronCount, U32 inputDataCount);

/** ニューラルネットワーククラスを作成する */
Layer::NeuralNetwork::INNLayerConnectData* CreateNeuralNetwork(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, U32 inputDataCount);

/** ニューラルネットワーククラスにレイヤーを追加する */
Gravisbell::ErrorCode CreateNeuralNetworkLayerConnect(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager,
	Layer::NeuralNetwork::INNLayerConnectData* pNeuralNetworkData,
	U32 outputDataCount,
	std::list<Layer::ILayerData*>& lppLayerData);

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

//	srand(12345);

	std::list<Layer::ILayerData*> lppLayerData;	// レイヤーデータの一覧

	// サンプルデータの読み込み
	Gravisbell::DataFormat::IDataFormatBase* pTeachData  = ::LoadSampleData(L"DataFormat.xml", L"../../SampleData/XOR/XOR.csv");
	printf("入力信号：%d\n", pTeachData->GetDataStruct(L"input").GetDataCount());
	printf("出力信号：%d\n", pTeachData->GetDataStruct(L"output").GetDataCount());

	// レイヤーDLL管理クラスを作成
	Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager = ::CreateLayerDLLManager();
	if(pLayerDLLManager == NULL)
	{
		delete pTeachData;
		return -1;
	}

	// 入出力信号レイヤーを作成
	Layer::IOData::IIODataLayer* pTeachInputLayer  = Layer::IOData::CreateIODataLayerCPU( pTeachData->GetDataStruct(L"input") );	// 入力信号(教師信号)
	Layer::IOData::IIODataLayer* pTeachOutputLayer = Layer::IOData::CreateIODataLayerCPU( pTeachData->GetDataStruct(L"output") );	// 出力信号(教師信号)
	for(U32 dataNum=0; dataNum<(U32)(pTeachData->GetDataCount()); dataNum++)
	{
		pTeachInputLayer->AddData(pTeachData->GetDataByNum(dataNum, L"input"));
		pTeachOutputLayer->AddData(pTeachData->GetDataByNum(dataNum, L"output"));

		printf("INPUT : %.3f, %.3f - OUTPUT : %.3f, %.3f, %.3f\n",
			pTeachData->GetDataByNum(dataNum, L"input")[0],  pTeachData->GetDataByNum(dataNum, L"input")[1],
			pTeachData->GetDataByNum(dataNum, L"output")[0], pTeachData->GetDataByNum(dataNum, L"output")[1], pTeachData->GetDataByNum(dataNum, L"output")[2]);
	}
	printf("訓練データ  ：%d\n", pTeachInputLayer->GetDataCount());

	// ニューラルネットワーククラスを作成
	Layer::NeuralNetwork::INNLayerConnectData* pNeuralNetworkData = ::CreateNeuralNetwork(*pLayerDLLManager, pTeachInputLayer->GetInputBufferCount());
	if(pNeuralNetworkData == NULL)
	{
		delete pTeachInputLayer;
		delete pTeachOutputLayer;
		delete pLayerDLLManager;
		delete pTeachData;
		return -1;
	}
	lppLayerData.push_back(pNeuralNetworkData);

	// ニューラルネットワークの接続を作成
	if(::CreateNeuralNetworkLayerConnect(*pLayerDLLManager, pNeuralNetworkData, pTeachOutputLayer->GetBufferCount(), lppLayerData) != ErrorCode::ERROR_CODE_NONE)
	{
		for(auto pLayerData : lppLayerData)
			delete pLayerData;
		delete pTeachInputLayer;
		delete pTeachOutputLayer;
		delete pLayerDLLManager;
		delete pTeachData;
		return -1;
	}

	// 学習,サンプル実行別実行
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
			for(auto pLayerData : lppLayerData)
				delete pLayerData;
			delete pTeachInputLayer;
			delete pTeachOutputLayer;
			delete pLayerDLLManager;
			delete pTeachData;
			return -1;
		}

		// 学習
		if(::LearnNeuralNetwork(pNeuralNetwork, pTeachInputLayer, pTeachOutputLayer, 8, 500) != ErrorCode::ERROR_CODE_NONE)
		{
			for(auto pLayerData : lppLayerData)
				delete pLayerData;
			delete pTeachInputLayer;
			delete pTeachOutputLayer;
			delete pLayerDLLManager;
			delete pTeachData;
			return -1;
		}

		// メモリ開放
		delete pNeuralNetwork;
	}
	
	// メモリ開放
	for(auto pLayerData : lppLayerData)
		delete pLayerData;
	delete pTeachInputLayer;
	delete pTeachOutputLayer;
	delete pLayerDLLManager;
	delete pTeachData;

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
/** レイヤーDLL管理クラスの作成 */
Layer::NeuralNetwork::ILayerDLLManager* CreateLayerDLLManager(void)
{
	// DLL管理クラスを作成
	Layer::NeuralNetwork::ILayerDLLManager* pDLLManager = Layer::NeuralNetwork::CreateLayerDLLManagerCPU();

	// DLLの読み込み.
	// FullyConnect
	{
	#ifdef _DEBUG
		if(pDLLManager->ReadLayerDLL(L"../../Debug/Gravisbell.Layer.NeuralNetwork.FullyConnect_Activation.dll") != Gravisbell::ErrorCode::ERROR_CODE_NONE)
	#else
		if(pDLLManager->ReadLayerDLL(L"../../Release/Gravisbell.Layer.NeuralNetwork.FullyConnect_Activation.dll") != Gravisbell::ErrorCode::ERROR_CODE_NONE)
	#endif
		{
			delete pDLLManager;
			return NULL;
		}
	}
	// Feedforward
	{
	#ifdef _DEBUG
		if(pDLLManager->ReadLayerDLL(L"../../Debug/Gravisbell.Layer.NeuralNetwork.FeedforwardNeuralNetwork.dll") != Gravisbell::ErrorCode::ERROR_CODE_NONE)
	#else
		if(pDLLManager->ReadLayerDLL(L"../../Release/Gravisbell.Layer.NeuralNetwork.FeedforwardNeuralNetwork.dll") != Gravisbell::ErrorCode::ERROR_CODE_NONE)
	#endif
		{
			delete pDLLManager;
			return NULL;
		}
	}

	return pDLLManager;
}


/** レイヤーデータを作成 */
Layer::NeuralNetwork::INNLayerData* CreateLayerData(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, U32 neuronCount, U32 inputDataCount, const std::wstring activationType)
{
	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(Gravisbell::GUID(0xbeba34ec, 0xc30c, 0x4565, 0x93, 0x86, 0x56, 0x08, 0x89, 0x81, 0xd2, 0xd7));
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;
	// ニューロン数
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"NeuronCount"));
		pItem->SetValue(neuronCount);
	}
	// 活性化関数種別
	{
		SettingData::Standard::IItem_Enum* pItem = dynamic_cast<SettingData::Standard::IItem_Enum*>(pConfig->GetItemByID(L"ActivationType"));
		pItem->SetValue(activationType.c_str());
//		pItem->SetValue(L"ReLU");
//		pItem->SetValue(L"sigmoid");
	}

	// レイヤーの作成
	Layer::NeuralNetwork::INNLayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, IODataStruct(inputDataCount));
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}
Layer::NeuralNetwork::INNLayerData* CreateHiddenLayerData(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, U32 neuronCount, U32 inputDataCount)
{
	return ::CreateLayerData(layerDLLManager, neuronCount, inputDataCount, L"ReLU");
//	return ::CreateLayerData(layerDLLManager, neuronCount, inputDataCount, L"sigmoid");
}
Layer::NeuralNetwork::INNLayerData* CreateOutputLayerData(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, U32 neuronCount, U32 inputDataCount)
{
//	return ::CreateLayerData(layerDLLManager, neuronCount, inputDataCount, L"ReLU");
	return ::CreateLayerData(layerDLLManager, neuronCount, inputDataCount, L"sigmoid");
}

/** ニューラルネットワーククラスを作成する */
Layer::NeuralNetwork::INNLayerConnectData* CreateNeuralNetwork(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, U32 inputDataCount)
{
	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(Gravisbell::GUID(0x1c38e21f, 0x6f01, 0x41b2, 0xb4, 0x0e, 0x7f, 0x67, 0x26, 0x7a, 0x36, 0x92));
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// レイヤーの作成
	Layer::NeuralNetwork::INNLayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, IODataStruct(inputDataCount));
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	// キャスト
	Layer::NeuralNetwork::INNLayerConnectData* pNeuralNetwork = dynamic_cast<Layer::NeuralNetwork::INNLayerConnectData*>(pLayer);
	if(pNeuralNetwork == NULL)
	{
		delete pLayer;
		return NULL;
	}

	return pNeuralNetwork;
}

/** ニューラルネットワーククラスにレイヤーを追加する */
Gravisbell::ErrorCode CreateNeuralNetworkLayerConnect(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager,
	Layer::NeuralNetwork::INNLayerConnectData* pNeuralNetworkData,
	U32 outputDataCount,
	std::list<Layer::ILayerData*>& lppLayerData)
{
	// 入力信号を直前レイヤーに設定
	Gravisbell::IODataStruct inputDataStruct = pNeuralNetworkData->GetInputDataStruct();
	Gravisbell::GUID lastLayerGUID = pNeuralNetworkData->GetInputGUID();

	// 1層目
	{
		// GUID生成
		Gravisbell::GUID guid = boost::uuids::random_generator()().data;

		// レイヤーデータを作成
		Layer::NeuralNetwork::INNLayerData* pLayerData = CreateHiddenLayerData(layerDLLManager, 128, inputDataStruct.GetDataCount());
		if(pLayerData == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
		lppLayerData.push_back(pLayerData);
		pNeuralNetworkData->AddLayer(guid, pLayerData);

		// 接続
		pNeuralNetworkData->AddInputLayerToLayer(guid, lastLayerGUID);

		// 現在レイヤーを直前レイヤーに変更
		inputDataStruct = pLayerData->GetOutputDataStruct();
		lastLayerGUID = guid;
	}

	// 2層目
	{
		// GUID生成
		Gravisbell::GUID guid = boost::uuids::random_generator()().data;

		// レイヤーデータを作成
		Layer::NeuralNetwork::INNLayerData* pLayerData = CreateHiddenLayerData(layerDLLManager, 128, inputDataStruct.GetDataCount());
		if(pLayerData == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
		lppLayerData.push_back(pLayerData);
		pNeuralNetworkData->AddLayer(guid, pLayerData);

		// 接続
		pNeuralNetworkData->AddInputLayerToLayer(guid, lastLayerGUID);

		// 現在レイヤーを直前レイヤーに変更
		inputDataStruct = pLayerData->GetOutputDataStruct();
		lastLayerGUID = guid;
	}

	//// 3層目
	//{
	//	// GUID生成
	//	Gravisbell::GUID guid = boost::uuids::random_generator()().data;

	//	// レイヤーデータを作成
	//	Layer::NeuralNetwork::INNLayerData* pLayerData = CreateHiddenLayerData(layerDLLManager, 128, inputDataStruct.GetDataCount());
	//	if(pLayerData == NULL)
	//		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	//	lppLayerData.push_back(pLayerData);
	//	pNeuralNetworkData->AddLayer(guid, pLayerData);

	//	// 接続
	//	pNeuralNetworkData->AddInputLayerToLayer(guid, lastLayerGUID);

	//	// 現在レイヤーを直前レイヤーに変更
	//	inputDataStruct = pLayerData->GetOutputDataStruct();
	//	lastLayerGUID = guid;
	//}

	// 4層目(出力層)
	{
		// GUID生成
		Gravisbell::GUID guid = boost::uuids::random_generator()().data;

		// レイヤーデータを作成
		Layer::NeuralNetwork::INNLayerData* pLayerData = CreateOutputLayerData(layerDLLManager, outputDataCount, inputDataStruct.GetDataCount());
		if(pLayerData == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
		lppLayerData.push_back(pLayerData);
		pNeuralNetworkData->AddLayer(guid, pLayerData);

		// 接続
		pNeuralNetworkData->AddInputLayerToLayer(guid, lastLayerGUID);

		// 現在レイヤーを直前レイヤーに変更
		inputDataStruct = pLayerData->GetOutputDataStruct();
		lastLayerGUID = guid;

		// 出力レイヤーに設定
		pNeuralNetworkData->SetOutputLayerGUID(guid);
	}

	return ErrorCode::ERROR_CODE_NONE;
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
		printf("%4d回 ", learnTime);

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
			pNeuralNetwork->CalculateLearnError(pTeachLayer->GetDInputBuffer());

			// 誤差を反映
			pNeuralNetwork->ReflectionLearnError();
		}

		// 誤差表示
		{
			F32 errorMin, errorMax, errorAve, errorAve2, errorCrossEntoropy;
			pTeachLayer->GetCalculateErrorValue(errorMin, errorMax, errorAve, errorAve2, errorCrossEntoropy);
			printf("min=%.3f, max=%.3f, ave=%.3f, ave2=%.3f\n", errorMin, errorMax, errorAve, errorAve2);
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
		F32 errorMin, errorMax, errorAve, errorAve2, errorCrossEntoropy;
		pTeachLayer->GetCalculateErrorValue(errorMin, errorMax, errorAve, errorAve2, errorCrossEntoropy);
		printf("min=%.3f, max=%.3f, ave=%.3f, ave2=%.3f\n", errorMin, errorMax, errorAve, errorAve2);
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
	pNeuralNetworkLearn->SetLearnSettingData(L"LearnCoeff", 0.1f);

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

	// 学習を実行
	for(U32 learnTime=0; learnTime<LEARN_TIMES; learnTime++)
	{
//		printf("%4d回 ", learnTime);
		printf("%4d,", learnTime);

		// 学習
		{
			// 学習ループ先頭処理
			pBatchDataNoListGenerator->PreProcessLearnLoop();
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
				pNeuralNetworkLearn->CalculateLearnError(pTeachOutputLayer->GetDInputBuffer());

				// 誤差を反映
				pNeuralNetworkLearn->ReflectionLearnError();
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
			}
		}

		// 誤差表示
		{
			F32 errorMin, errorMax, errorAve, errorAve2, errorCrossEntoropy;
			pTeachOutputLayer->GetCalculateErrorValue(errorMin, errorMax, errorAve, errorAve2, errorCrossEntoropy);
//			printf("学習：min=%.3f, max=%.3f, ave=%.3f, ave2=%.3f", errorMin, errorMax, errorAve, errorAve2);
			printf("%.3f,%.3f,%.3f,%.3f,", errorMin, errorMax, errorAve, errorAve2);
		}
//		printf(" : ");
		{
			F32 errorMin, errorMax, errorAve, errorAve2, errorCrossEntoropy;
			pSampleOutputLayer->GetCalculateErrorValue(errorMin, errorMax, errorAve, errorAve2, errorCrossEntoropy);
//			printf("実行：min=%.3f, max=%.3f, ave=%.3f, ave2=%.3f", errorMin, errorMax, errorAve, errorAve2);
			printf("%.3f,%.3f,%.3f,%.3f,", errorMin, errorMax, errorAve, errorAve2);
		}
		printf("\n");
	}

	// メモリ開放
	delete pLearnSetting;
	delete pBatchDataNoListGenerator;

	return ErrorCode::ERROR_CODE_NONE;
}