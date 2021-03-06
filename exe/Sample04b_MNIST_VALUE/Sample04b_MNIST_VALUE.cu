// Sample04b_MNIST_VALUE.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include<crtdbg.h>

#include<cuda.h>
#include<cuda_runtime.h>
#include<thrust/device_vector.h>

#include<vector>
#include<boost/filesystem/path.hpp>
#include<boost/uuid/uuid_generators.hpp>

#include"Library/Common/BatchDataNoListGenerator.h"
#include"Library/DataFormat/Binary.h"
#include"Library/NeuralNetwork/LayerDLLManager.h"
#include"Library/NeuralNetwork/LayerDataManager.h"
#include"Library/NeuralNetwork/NetworkParserXML.h"
#include"Library/Layer/IOData/IODataLayer.h"
#include"Layer/Connect/ILayerConnectData.h"
#include"Layer/NeuralNetwork/INeuralNetwork.h"
#include"Utility/NeuralNetworkLayer.h"
#include"Utility/NeuralNetworkMaker.h"

#include"Library/NeuralNetwork/Initializer.h"

using namespace Gravisbell;

#define USE_GPU	1
#define USE_HOST_MEMORY 1

#define USE_BATCHNORM	1
#define USE_DROPOUT		1

#define USE_BATCH_SIZE	64
#define MAX_EPOCH		5


/** データファイルをを読み込む
	@param	o_ppDataLayerTeach	教師データを格納したデータクラスの格納先ポインタアドレス
	@param	o_ppDataLayerTest	テストデータを格納したデータクラスの格納先ポインタアドレス
	@param	i_testRate			テストデータを全体の何%にするか0〜1の間で設定
	@param	i_formatFilePath	フォーマット設定の入ったXMLファイルパス
	@param	i_dataFilePath		データの入ったバイナリファイルパス
	*/
Gravisbell::ErrorCode LoadSampleData_image(
	Layer::IOData::IIODataLayer** o_ppDataLayerTeach,  Layer::IOData::IIODataLayer** o_ppDataLayerTest,
	F32 i_testRate,
	boost::filesystem::wpath i_formatFilePath,
	boost::filesystem::wpath i_dataFilePath);
Gravisbell::ErrorCode LoadSampleData_label(
	Layer::IOData::IIODataLayer** o_ppDataLayerTeach,  Layer::IOData::IIODataLayer** o_ppDataLayerTest,
	F32 i_testRate,
	boost::filesystem::wpath i_formatFilePath,
	boost::filesystem::wpath i_dataFilePath);

/** ニューラルネットワーククラスを作成する */
Layer::Connect::ILayerConnectData* CreateNeuralNetwork(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct& inputDataStruct, const IODataStruct& outputDataStruct);

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

	//void* pValue = NULL;
	//cudaMalloc(&pValue, 16);
	//cudaFree(&pValue);

	boost::filesystem::path workDirPath = boost::filesystem::current_path();

	// 画像を読み込み
	Layer::IOData::IIODataLayer* pDataLayerTeach_Input  = NULL;
	Layer::IOData::IIODataLayer* pDataLayerTeach_Output = NULL;
	Layer::IOData::IIODataLayer* pDataLayerTest_Input  = NULL;
	Layer::IOData::IIODataLayer* pDataLayerTest_Output = NULL;

#ifndef _WIN64
	if(LoadSampleData_image(&pDataLayerTeach_Input, &pDataLayerTest_Input, 0.1f, L"../SampleData/MNIST/DataFormat_image.xml", L"../SampleData/MNIST/train-images.idx3-ubyte") != Gravisbell::ErrorCode::ERROR_CODE_NONE)
#else
	if(LoadSampleData_image(&pDataLayerTeach_Input, &pDataLayerTest_Input, 0.1f, L"../../SampleData/MNIST/DataFormat_image.xml", L"../../SampleData/MNIST/train-images.idx3-ubyte") != Gravisbell::ErrorCode::ERROR_CODE_NONE)
#endif
	{
		return -1;
	}
#ifndef _WIN64
	if(LoadSampleData_label(&pDataLayerTeach_Output, &pDataLayerTest_Output, 0.1f, L"../SampleData/MNIST/DataFormat_label.xml", L"../SampleData/MNIST/train-labels.idx1-ubyte") != Gravisbell::ErrorCode::ERROR_CODE_NONE)
#else
	if(LoadSampleData_label(&pDataLayerTeach_Output, &pDataLayerTest_Output, 0.1f, L"../../SampleData/MNIST/DataFormat_label.xml", L"../../SampleData/MNIST/train-labels.idx1-ubyte") != Gravisbell::ErrorCode::ERROR_CODE_NONE)
#endif
	{
		delete pDataLayerTeach_Input;
		delete pDataLayerTest_Input;

		return -1;
	}

	workDirPath = boost::filesystem::current_path();

	// レイヤーDLL管理クラスを作成
#if USE_GPU
	Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager = Gravisbell::Utility::NeuralNetworkLayer::CreateLayerDLLManagerGPU(L"./");
#else
	Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager = Gravisbell::Utility::NeuralNetworkLayer::CreateLayerDLLManagerCPU(L"./");
#endif
	if(pLayerDLLManager == NULL)
	{
		delete pDataLayerTeach_Input;
		delete pDataLayerTeach_Output;
		delete pDataLayerTest_Input;
		delete pDataLayerTest_Output;
		return -1;
	}

	// レイヤーデータ管理クラスを作成
	Gravisbell::Layer::NeuralNetwork::ILayerDataManager* pLayerDataManager = Gravisbell::Layer::NeuralNetwork::CreateLayerDataManager();
	if(pLayerDataManager == NULL)
	{
		delete pDataLayerTeach_Input;
		delete pDataLayerTeach_Output;
		delete pDataLayerTest_Input;
		delete pDataLayerTest_Output;
		delete pLayerDLLManager;
		return -1;
	}

	// 乱数を固定
//#ifdef _DEBUG
	Gravisbell::Layer::NeuralNetwork::GetInitializerManager().InitializeRandomParameter(0);
//#endif

	// ニューラルネットワーク作成
	Gravisbell::Layer::Connect::ILayerConnectData* pNeuralNetworkData = CreateNeuralNetwork(*pLayerDLLManager, *pLayerDataManager, pDataLayerTeach_Input->GetInputDataStruct(), pDataLayerTeach_Output->GetDataStruct());
	if(pNeuralNetworkData == NULL)
	{
		delete pDataLayerTeach_Input;
		delete pDataLayerTeach_Output;
		delete pDataLayerTest_Input;
		delete pDataLayerTest_Output;
		delete pLayerDataManager;
		delete pLayerDLLManager;
		return -1;
	}


	// ファイルに保存する
	printf("バイナリファイル保存\n");
	Gravisbell::Utility::NeuralNetworkLayer::WriteNetworkToBinaryFile(*pNeuralNetworkData, L"../../LayerData/test.bin");
	// ファイルから読み込む
	{
		pLayerDataManager->EraseLayerByGUID(pNeuralNetworkData->GetGUID());
		pNeuralNetworkData = NULL;
		Gravisbell::Layer::ILayerData* pTmpNeuralNetworkData = NULL;

		printf("バイナリファイル読み込み\n");
		Gravisbell::Utility::NeuralNetworkLayer::ReadNetworkFromBinaryFile(*pLayerDLLManager, &pTmpNeuralNetworkData,  L"../../LayerData/test.bin");
		// 別ファイルに保存する
		printf("バイナリファイル保存2\n");
		Gravisbell::Utility::NeuralNetworkLayer::WriteNetworkToBinaryFile(*pTmpNeuralNetworkData,  L"../../LayerData/test2.bin");
		printf("終了\n");

		pNeuralNetworkData = dynamic_cast<Gravisbell::Layer::Connect::ILayerConnectData*>(pTmpNeuralNetworkData);
	}

	//// XMLファイルに保存する
	//Gravisbell::Layer::NeuralNetwork::Parser::SaveLayerToXML(*pNeuralNetworkData, L"../../LayerData/", L"test.xml");
	//// ファイルから読み込む
	//for(auto pLayerData : lppLayerData)
	//	delete pLayerData;
	//lppLayerData.clear();
	//pNeuralNetworkData = Gravisbell::Layer::NeuralNetwork::Parser::CreateLayerFromXML(*pLayerDLLManager, *pLayerDataManager, L"../../LayerData/", L"test.xml");
	//// バイナリファイルに保存する
	//Gravisbell::Utility::NeuralNetworkLayer::WriteNetworkToBinaryFile(*pNeuralNetworkData, "../../LayerData/test2.bin");
	//// 別のXMLファイルに保存する
	//Gravisbell::Layer::NeuralNetwork::Parser::SaveLayerToXML(*pNeuralNetworkData, L"../../LayerData/", L"test2.xml");


	// 学習用ニューラルネットワーク作成
	Layer::NeuralNetwork::INeuralNetwork* pNeuralNetworkLearn = NULL;
	{
#if USE_HOST_MEMORY
		Layer::ILayerBase* pLayer = pNeuralNetworkData->CreateLayer(boost::uuids::random_generator()().data, &pDataLayerTeach_Input->GetOutputDataStruct(), 1);
#else
		Layer::ILayerBase* pLayer = pNeuralNetworkData->CreateLayer_device(boost::uuids::random_generator()().data, &pDataLayerTeach_Input->GetOutputDataStruct(), 1);
#endif
		pNeuralNetworkLearn = dynamic_cast<Layer::NeuralNetwork::INeuralNetwork*>(pLayer);
		if(pNeuralNetworkLearn == NULL)
		{
			if(pLayer)
				delete pLayer;
		}
	}
	if(pNeuralNetworkLearn == NULL)
	{
		delete pDataLayerTeach_Input;
		delete pDataLayerTeach_Output;
		delete pDataLayerTest_Input;
		delete pDataLayerTest_Output;
		delete pLayerDataManager;
		delete pLayerDLLManager;
		return -1;
	}

	// テスト用ニューラルネットワーク作成
	Layer::NeuralNetwork::INeuralNetwork* pNeuralNetworkTest = NULL;
	{
#if USE_HOST_MEMORY
		Layer::ILayerBase* pLayer = pNeuralNetworkData->CreateLayer(boost::uuids::random_generator()().data, &pDataLayerTeach_Input->GetOutputDataStruct(), 1);
#else
		Layer::ILayerBase* pLayer = pNeuralNetworkData->CreateLayer_device(boost::uuids::random_generator()().data, &pDataLayerTeach_Input->GetOutputDataStruct(), 1);
#endif
		pNeuralNetworkTest = dynamic_cast<Layer::NeuralNetwork::INeuralNetwork*>(pLayer);
		if(pNeuralNetworkTest == NULL)
		{
			if(pLayer)
				delete pLayer;
		}
	}
	if(pNeuralNetworkTest == NULL)
	{
		delete pNeuralNetworkLearn;
		delete pDataLayerTeach_Input;
		delete pDataLayerTeach_Output;
		delete pDataLayerTest_Input;
		delete pDataLayerTest_Output;
		delete pLayerDataManager;
		delete pLayerDLLManager;
		return -1;
	}

	// 学習, テスト実行
	{
		time_t startTime = time(NULL);

		// 学習
		if(::LearnWithCalculateSampleError(pNeuralNetworkLearn, pNeuralNetworkTest, pDataLayerTeach_Input, pDataLayerTeach_Output, pDataLayerTest_Input, pDataLayerTest_Output, USE_BATCH_SIZE, MAX_EPOCH) != ErrorCode::ERROR_CODE_NONE)
		{
			delete pNeuralNetworkLearn;
			delete pNeuralNetworkTest;
			delete pDataLayerTeach_Input;
			delete pDataLayerTeach_Output;
			delete pDataLayerTest_Input;
			delete pDataLayerTest_Output;
			delete pLayerDataManager;
			delete pLayerDLLManager;

			return -1;
		}

		time_t endTime = time(NULL);

		printf("経過時間(s) : %ld\n", (endTime - startTime));
	}


	// バッファ開放
	delete pNeuralNetworkData;
	delete pNeuralNetworkLearn;
	delete pNeuralNetworkTest;
	delete pDataLayerTeach_Input;
	delete pDataLayerTeach_Output;
	delete pDataLayerTest_Input;
	delete pDataLayerTest_Output;
	delete pLayerDataManager;
	delete pLayerDLLManager;

	printf("Press any key to continue");
	getc(stdin);


	return 0;
}


/** データファイルをを読み込む
	@param	o_ppDataLayerTeach	教師データを格納したデータクラスの格納先ポインタアドレス
	@param	o_ppDataLayerTest	テストデータを格納したデータクラスの格納先ポインタアドレス
	@param	i_testRate			テストデータを全体の何%にするか0〜1の間で設定
	@param	i_formatFilePath	フォーマット設定の入ったXMLファイルパス
	@param	i_dataFilePath		データの入ったバイナリファイルパス
	*/
Gravisbell::ErrorCode LoadSampleData_image(
	Layer::IOData::IIODataLayer** o_ppDataLayerTeach,  Layer::IOData::IIODataLayer** o_ppDataLayerTest,
	F32 i_testRate,
	boost::filesystem::wpath i_formatFilePath,
	boost::filesystem::wpath i_dataFilePath)
{
	// フォーマットを読み込む
	Gravisbell::DataFormat::Binary::IDataFormat* pDataFormat = Gravisbell::DataFormat::Binary::CreateDataFormatFromXML(i_formatFilePath.c_str());
	if(pDataFormat == NULL)
		return Gravisbell::ErrorCode::ERROR_CODE_COMMON_FILE_NOT_FOUND;

	// バッファを読み込む
	std::vector<BYTE> lpBuf;
	{
		FILE* fp = _wfopen(i_dataFilePath.c_str(), L"rb");
		if(fp == NULL)
		{
			delete pDataFormat;
			return Gravisbell::ErrorCode::ERROR_CODE_COMMON_FILE_NOT_FOUND;
		}

		fseek(fp, 0, SEEK_END);
		U32 fileSize = ftell(fp);
		lpBuf.resize(fileSize);

		fseek(fp, 0, SEEK_SET);
		fread(&lpBuf[0], 1, fileSize, fp);

		fclose(fp);
	}

	// フォーマットを使ってヘッダを読み込む
	U32 bufPos = 0;

	// ヘッダを読み込む
	bufPos = pDataFormat->LoadBinary(&lpBuf[0], (U32)lpBuf.size());

	// データ構造を作成する
	Gravisbell::IODataStruct dataStruct(1, pDataFormat->GetVariableValue(L"columns"), pDataFormat->GetVariableValue(L"rows"), 1);


#if USE_GPU
#if USE_HOST_MEMORY
	*o_ppDataLayerTeach = Gravisbell::Layer::IOData::CreateIODataLayerCPU(dataStruct);
	*o_ppDataLayerTest  = Gravisbell::Layer::IOData::CreateIODataLayerCPU(dataStruct);
#else
	*o_ppDataLayerTeach = Gravisbell::Layer::IOData::CreateIODataLayerGPU_host(dataStruct);
	*o_ppDataLayerTest  = Gravisbell::Layer::IOData::CreateIODataLayerGPU_host(dataStruct);
	//*o_ppDataLayerTeach = Gravisbell::Layer::IOData::CreateIODataLayerGPU_device(dataStruct);
	//*o_ppDataLayerTest  = Gravisbell::Layer::IOData::CreateIODataLayerGPU_device(dataStruct);
#endif
#else
	*o_ppDataLayerTeach = Gravisbell::Layer::IOData::CreateIODataLayerCPU(dataStruct);
	*o_ppDataLayerTest  = Gravisbell::Layer::IOData::CreateIODataLayerCPU(dataStruct);
#endif


	std::vector<F32> lpTmpBuf(dataStruct.GetDataCount());

	// データの見込み
	U32 dataCount = (U32)pDataFormat->GetVariableValue(L"images");
	U32 teachDataCount = (U32)(dataCount*(1.0f - i_testRate));
	for(U32 imageNum=0; imageNum<dataCount; imageNum++)
	{
		if(bufPos + dataStruct.GetDataCount() > lpBuf.size())
			break;

		// U08 -> F32 変換
		for(U32 i=0; i<lpTmpBuf.size(); i++)
		{
			lpTmpBuf[i] = (F32)lpBuf[bufPos + i] / 0xFF;
		}

		if(imageNum < teachDataCount)
			(*o_ppDataLayerTeach)->AddData(&lpTmpBuf[0]);
		else
			(*o_ppDataLayerTest)->AddData(&lpTmpBuf[0]);

		bufPos += dataStruct.GetDataCount();
	}

	// データフォーマット削除
	delete pDataFormat;

	return Gravisbell::ErrorCode::ERROR_CODE_NONE;
}
Gravisbell::ErrorCode LoadSampleData_label(
	Layer::IOData::IIODataLayer** o_ppDataLayerTeach,  Layer::IOData::IIODataLayer** o_ppDataLayerTest,
	F32 i_testRate,
	boost::filesystem::wpath i_formatFilePath,
	boost::filesystem::wpath i_dataFilePath)
{
	// フォーマットを読み込む
	Gravisbell::DataFormat::Binary::IDataFormat* pDataFormat = Gravisbell::DataFormat::Binary::CreateDataFormatFromXML(i_formatFilePath.c_str());
	if(pDataFormat == NULL)
		return Gravisbell::ErrorCode::ERROR_CODE_COMMON_FILE_NOT_FOUND;

	// バッファを読み込む
	std::vector<BYTE> lpBuf;
	{
		FILE* fp = _wfopen(i_dataFilePath.c_str(), L"rb");
		if(fp == NULL)
		{
			delete pDataFormat;
			return Gravisbell::ErrorCode::ERROR_CODE_COMMON_FILE_NOT_FOUND;
		}

		fseek(fp, 0, SEEK_END);
		U32 fileSize = ftell(fp);
		lpBuf.resize(fileSize);

		fseek(fp, 0, SEEK_SET);
		fread(&lpBuf[0], 1, fileSize, fp);

		fclose(fp);
	}

	// フォーマットを使ってヘッダを読み込む
	U32 bufPos = 0;

	// ヘッダを読み込む
	bufPos = pDataFormat->LoadBinary(&lpBuf[0], (U32)lpBuf.size());

	// データ構造を作成する
	Gravisbell::IODataStruct dataStruct(1, 1, 1, 1);


#if USE_GPU
#if USE_HOST_MEMORY
	*o_ppDataLayerTeach = Gravisbell::Layer::IOData::CreateIODataLayerCPU(dataStruct);
	*o_ppDataLayerTest  = Gravisbell::Layer::IOData::CreateIODataLayerCPU(dataStruct);
#else
	*o_ppDataLayerTeach = Gravisbell::Layer::IOData::CreateIODataLayerGPU_host(dataStruct);
	*o_ppDataLayerTest  = Gravisbell::Layer::IOData::CreateIODataLayerGPU_host(dataStruct);
//	*o_ppDataLayerTeach = Gravisbell::Layer::IOData::CreateIODataLayerGPU_device(dataStruct);
//	*o_ppDataLayerTest  = Gravisbell::Layer::IOData::CreateIODataLayerGPU_device(dataStruct);
#endif
#else
	*o_ppDataLayerTeach = Gravisbell::Layer::IOData::CreateIODataLayerCPU(dataStruct);
	*o_ppDataLayerTest  = Gravisbell::Layer::IOData::CreateIODataLayerCPU(dataStruct);
#endif


	// データの見込み
	U32 dataCount = (U32)pDataFormat->GetVariableValue(L"images");
	U32 teachDataCount = (U32)(dataCount*(1.0f - i_testRate));
	for(U32 imageNum=0; imageNum<dataCount; imageNum++)
	{
		F32 value = (F32)lpBuf[bufPos] / 9.0f * 2.0f - 1;	// 最大値を1.0にする必要があるため、10ではなく9で割る

		if(imageNum < teachDataCount)
			(*o_ppDataLayerTeach)->AddData(&value);
		else
			(*o_ppDataLayerTest)->AddData(&value);

		bufPos += 1;
	}

	// データフォーマット削除
	delete pDataFormat;

	return Gravisbell::ErrorCode::ERROR_CODE_NONE;
}


Layer::Connect::ILayerConnectData* CreateNeuralNetwork_ver01(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
{
	// ニューラルネットワーク作成クラスを作成
	Gravisbell::Utility::NeuralNetworkLayer::INeuralNetworkMaker* pNetworkMaker = Gravisbell::Utility::NeuralNetworkLayer::CreateNeuralNetworkManaker(layerDLLManager, layerDataManager, &i_inputDataStruct, 1);

	// ニューラルネットワークを作成
	Layer::Connect::ILayerConnectData* pNeuralNetwork = pNetworkMaker->GetNeuralNetworkLayer();
	if(pNeuralNetwork == NULL)
		return NULL;


	// レイヤーを追加する
	if(pNeuralNetwork)
	{
		// 入力信号を直前レイヤーに設定
		Gravisbell::GUID lastLayerGUID = pNeuralNetwork->GetInputGUID(0);

		lastLayerGUID = pNetworkMaker->AddConvolutionLayer(lastLayerGUID, Vector3D<S32>(3,3,1), 8, Vector3D<S32>(1,1,1), Vector3D<S32>(1,1,0));
//		lastLayerGUID = pNetworkMaker->AddDilatedConvolutionLayer(lastLayerGUID, Vector3D<S32>(3,3,1), 8, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,0));
		lastLayerGUID = pNetworkMaker->AddActivationLayer(lastLayerGUID, L"ReLU");
		lastLayerGUID = pNetworkMaker->AddConvolutionLayer(lastLayerGUID, Vector3D<S32>(3,3,1), 16, Vector3D<S32>(1,1,1), Vector3D<S32>(1,1,0));
//		lastLayerGUID = pNetworkMaker->AddDilatedConvolutionLayer(lastLayerGUID, Vector3D<S32>(3,3,1), 16, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1), Vector3D<S32>(0,0,0));
		lastLayerGUID = pNetworkMaker->AddActivationLayer(lastLayerGUID, L"ReLU");

		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, 10, L"softmax_ALL_crossEntropy");
		lastLayerGUID = pNetworkMaker->AddSignalArray2ValueLayer(lastLayerGUID, -1.0, 1.0f);

		// 出力レイヤー設定
		pNeuralNetwork->SetOutputLayerGUID(lastLayerGUID);
	}

	// 出力データ構造が正しいことを確認
	if(pNeuralNetwork->GetOutputDataStruct(&i_inputDataStruct, 1) != i_outputDataStruct)
	{
		layerDataManager.EraseLayerByGUID(pNeuralNetwork->GetGUID());
		return NULL;
	}


	// オプティマイザーの設定
	pNeuralNetwork->ChangeOptimizer(L"Adam");

	delete pNetworkMaker;

	return pNeuralNetwork;
}

Layer::Connect::ILayerConnectData* CreateNeuralNetwork_ver02(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
{
	// ニューラルネットワーク作成クラスを作成
	Gravisbell::Utility::NeuralNetworkLayer::INeuralNetworkMaker* pNetworkMaker = Gravisbell::Utility::NeuralNetworkLayer::CreateNeuralNetworkManaker(layerDLLManager, layerDataManager, &i_inputDataStruct, 1);

	// ニューラルネットワークを作成
	Layer::Connect::ILayerConnectData* pNeuralNetwork = pNetworkMaker->GetNeuralNetworkLayer();
	if(pNeuralNetwork == NULL)
		return NULL;


	// レイヤーを追加する
	if(pNeuralNetwork)
	{
		// 入力信号を直前レイヤーに設定
		Gravisbell::GUID lastLayerGUID = pNeuralNetwork->GetInputGUID(0);

		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, (31*4 + 1)*4, L"ReLU");
		lastLayerGUID = pNetworkMaker->AddReshapeLayer(lastLayerGUID, 4, 31*4+1, 1, 1);

		std::vector<IODataStruct> lpTmpDataStruct;
		for(U32 layerNum=0; layerNum<4; layerNum++)
		{
			Gravisbell::GUID layerGUID_bypass = lastLayerGUID;
			Gravisbell::GUID layerGUID_tanh = lastLayerGUID;
			Gravisbell::GUID layerGUID_sigmoid = lastLayerGUID;
			for(U32 i=0; i<5; i++)
			{
				U32 delay = (U32)pow(2, i);
				layerGUID_tanh = pNetworkMaker->AddDilatedConvolutionLayer(layerGUID_tanh, Vector3D<S32>(2,1,1), 16, Vector3D<S32>( delay,1,1), Vector3D<S32>(1,1,1), Vector3D<S32>(0,0,0));
				if(i != 4)
					layerGUID_tanh = pNetworkMaker->AddActivationLayer(layerGUID_tanh, L"ReLU");
			}
			for(U32 i=0; i<5; i++)
			{
				U32 delay = (U32)pow(2, i);
				layerGUID_sigmoid = pNetworkMaker->AddDilatedConvolutionLayer(layerGUID_sigmoid, Vector3D<S32>(2,1,1), 16, Vector3D<S32>( delay,1,1), Vector3D<S32>(1,1,1), Vector3D<S32>(0,0,0));
				if(i != 4)
					layerGUID_sigmoid = pNetworkMaker->AddActivationLayer(layerGUID_sigmoid, L"ReLU");
			}

			layerGUID_tanh    = pNetworkMaker->AddActivationLayer(layerGUID_tanh, L"tanh");
			layerGUID_sigmoid = pNetworkMaker->AddActivationLayer(layerGUID_sigmoid, L"sigmoid");

			lastLayerGUID = pNetworkMaker->AddMergeMultiplyLayer(Utility::NeuralNetworkLayer::LayerMergeType::LYAERMERGETYPE_LAYER0, layerGUID_tanh, layerGUID_sigmoid);
			lastLayerGUID = pNetworkMaker->AddConvolutionLayer(lastLayerGUID, Vector3D<S32>(1,1,1), 16, Vector3D<S32>(1,1,1), Vector3D<S32>(0,0,0));

			IODataStruct tmpDataStruct = pNetworkMaker->GetOutputDataStruct(lastLayerGUID);
			layerGUID_bypass = pNetworkMaker->AddChooseBoxLayer(layerGUID_bypass, Vector3D<S32>(0,0,0), Vector3D<S32>(tmpDataStruct.x, tmpDataStruct.y, tmpDataStruct.z));

			lastLayerGUID = pNetworkMaker->AddMergeAddLayer(Utility::NeuralNetworkLayer::LayerMergeType::LYAERMERGETYPE_LAYER0, sqrtf(0.5f), lastLayerGUID, layerGUID_bypass);
		}
		IODataStruct tmpDataStruct2 = pNetworkMaker->GetOutputDataStruct(lastLayerGUID);

		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, 10, L"softmax_ALL");
		lastLayerGUID = pNetworkMaker->AddSignalArray2ValueLayer(lastLayerGUID, 0.0f, 1.0f);

		// 出力レイヤー設定
		pNeuralNetwork->SetOutputLayerGUID(lastLayerGUID);
	}

	// 出力データ構造が正しいことを確認
	if(pNeuralNetwork->GetOutputDataStruct(&i_inputDataStruct, 1) != i_outputDataStruct)
	{
		layerDataManager.EraseLayerByGUID(pNeuralNetwork->GetGUID());
		return NULL;
	}


	// オプティマイザーの設定
	pNeuralNetwork->ChangeOptimizer(L"Adam");

	delete pNetworkMaker;

	return pNeuralNetwork;
}

Layer::Connect::ILayerConnectData* CreateNeuralNetwork_ver03(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
{
	// ニューラルネットワーク作成クラスを作成
	Gravisbell::Utility::NeuralNetworkLayer::INeuralNetworkMaker* pNetworkMaker = Gravisbell::Utility::NeuralNetworkLayer::CreateNeuralNetworkManaker(layerDLLManager, layerDataManager, &i_inputDataStruct, 1);

	// ニューラルネットワークを作成
	Layer::Connect::ILayerConnectData* pNeuralNetwork = pNetworkMaker->GetNeuralNetworkLayer();
	if(pNeuralNetwork == NULL)
		return NULL;


	// レイヤーを追加する
	if(pNeuralNetwork)
	{
		// 入力信号を直前レイヤーに設定
		Gravisbell::GUID lastLayerGUID = pNeuralNetwork->GetInputGUID(0);

		lastLayerGUID = pNetworkMaker->AddConvolutionLayer(lastLayerGUID, Vector3D<S32>(3,3,1), 8, Vector3D<S32>(1,1,1), Vector3D<S32>(1,1,0));
//		lastLayerGUID = pNetworkMaker->AddDilatedConvolutionLayer(lastLayerGUID, Vector3D<S32>(3,3,1), 8, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,0));
		lastLayerGUID = pNetworkMaker->AddActivationLayer(lastLayerGUID, L"ReLU");
		lastLayerGUID = pNetworkMaker->AddConvolutionLayer(lastLayerGUID, Vector3D<S32>(3,3,1), 16, Vector3D<S32>(1,1,1), Vector3D<S32>(1,1,0));
//		lastLayerGUID = pNetworkMaker->AddDilatedConvolutionLayer(lastLayerGUID, Vector3D<S32>(3,3,1), 16, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1), Vector3D<S32>(0,0,0));
		lastLayerGUID = pNetworkMaker->AddActivationLayer(lastLayerGUID, L"ReLU");

		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, 10, L"softmax_ALL_crossEntropy");
//		lastLayerGUID = pNetworkMaker->AddProbabilityArray2ValueLayer(lastLayerGUID, -1.0, 1.0f, 0.1f);
		lastLayerGUID = pNetworkMaker->AddSignalArray2ValueLayer(lastLayerGUID, -1.0f, 1.0f);

		// 出力レイヤー設定
		pNeuralNetwork->SetOutputLayerGUID(lastLayerGUID);
	}

	// 出力データ構造が正しいことを確認
	if(pNeuralNetwork->GetOutputDataStruct(&i_inputDataStruct, 1) != i_outputDataStruct)
	{
		layerDataManager.EraseLayerByGUID(pNeuralNetwork->GetGUID());
		return NULL;
	}


	// オプティマイザーの設定
	pNeuralNetwork->ChangeOptimizer(L"Adam");

	delete pNetworkMaker;

	return pNeuralNetwork;
}

Layer::Connect::ILayerConnectData* CreateNeuralNetwork_ver04(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
{
	// ニューラルネットワーク作成クラスを作成
	Gravisbell::Utility::NeuralNetworkLayer::INeuralNetworkMaker* pNetworkMaker = Gravisbell::Utility::NeuralNetworkLayer::CreateNeuralNetworkManaker(layerDLLManager, layerDataManager, &i_inputDataStruct, 1);

	// ニューラルネットワークを作成
	Layer::Connect::ILayerConnectData* pNeuralNetwork = pNetworkMaker->GetNeuralNetworkLayer();
	if(pNeuralNetwork == NULL)
		return NULL;


	// レイヤーを追加する
	if(pNeuralNetwork)
	{
		// 入力信号を直前レイヤーに設定
		Gravisbell::GUID lastLayerGUID = pNeuralNetwork->GetInputGUID(0);

		// 入力信号をN段階のフラグに置き換え
		lastLayerGUID = pNetworkMaker->AddValue2SignalArrayLayer(lastLayerGUID, 0.0f, 1.0f, 32);

		lastLayerGUID = pNetworkMaker->AddConvolutionLayer(lastLayerGUID, Vector3D<S32>(3,3,1), 8, Vector3D<S32>(1,1,1), Vector3D<S32>(1,1,0));
//		lastLayerGUID = pNetworkMaker->AddDilatedConvolutionLayer(lastLayerGUID, Vector3D<S32>(3,3,1), 8, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,0));
		lastLayerGUID = pNetworkMaker->AddActivationLayer(lastLayerGUID, L"ReLU");
		lastLayerGUID = pNetworkMaker->AddConvolutionLayer(lastLayerGUID, Vector3D<S32>(3,3,1), 16, Vector3D<S32>(1,1,1), Vector3D<S32>(1,1,0));
//		lastLayerGUID = pNetworkMaker->AddDilatedConvolutionLayer(lastLayerGUID, Vector3D<S32>(3,3,1), 16, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1), Vector3D<S32>(0,0,0));
		lastLayerGUID = pNetworkMaker->AddActivationLayer(lastLayerGUID, L"ReLU");

		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, 10, L"softmax_ALL_crossEntropy");
//		lastLayerGUID = pNetworkMaker->AddProbabilityArray2ValueLayer(lastLayerGUID, -1.0, 1.0f, 0.1f);
		lastLayerGUID = pNetworkMaker->AddSignalArray2ValueLayer(lastLayerGUID, -1.0f, 1.0f);

		// 出力レイヤー設定
		pNeuralNetwork->SetOutputLayerGUID(lastLayerGUID);
	}

	// 出力データ構造が正しいことを確認
	if(pNeuralNetwork->GetOutputDataStruct(&i_inputDataStruct, 1) != i_outputDataStruct)
	{
		layerDataManager.EraseLayerByGUID(pNeuralNetwork->GetGUID());
		return NULL;
	}


	// オプティマイザーの設定
	pNeuralNetwork->ChangeOptimizer(L"Adam");

	delete pNetworkMaker;

	return pNeuralNetwork;
}

Layer::Connect::ILayerConnectData* CreateNeuralNetwork_ver05(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
{
	// ニューラルネットワーク作成クラスを作成
	Gravisbell::Utility::NeuralNetworkLayer::INeuralNetworkMaker* pNetworkMaker = Gravisbell::Utility::NeuralNetworkLayer::CreateNeuralNetworkManaker(layerDLLManager, layerDataManager, &i_inputDataStruct, 1);

	// ニューラルネットワークを作成
	Layer::Connect::ILayerConnectData* pNeuralNetwork = pNetworkMaker->GetNeuralNetworkLayer();
	if(pNeuralNetwork == NULL)
		return NULL;


	// レイヤーを追加する
	if(pNeuralNetwork)
	{
		// 入力信号を直前レイヤーに設定
		Gravisbell::GUID lastLayerGUID = pNeuralNetwork->GetInputGUID(0);

		// 入力信号をN段階のフラグに置き換え
		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_CA(lastLayerGUID, Vector3D<S32>(3,3,1), 4, Vector3D<S32>(1,1,1), Vector3D<S32>(1,1,0), L"sigmoid");
		lastLayerGUID = pNetworkMaker->AddValue2SignalArrayLayer(lastLayerGUID, 0.0f, 1.0f, 16);

		lastLayerGUID = pNetworkMaker->AddConvolutionLayer(lastLayerGUID, Vector3D<S32>(3,3,1), 8, Vector3D<S32>(1,1,1), Vector3D<S32>(1,1,0));
//		lastLayerGUID = pNetworkMaker->AddDilatedConvolutionLayer(lastLayerGUID, Vector3D<S32>(3,3,1), 8, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,0));
		lastLayerGUID = pNetworkMaker->AddActivationLayer(lastLayerGUID, L"ReLU");
		lastLayerGUID = pNetworkMaker->AddConvolutionLayer(lastLayerGUID, Vector3D<S32>(3,3,1), 16, Vector3D<S32>(1,1,1), Vector3D<S32>(1,1,0));
//		lastLayerGUID = pNetworkMaker->AddDilatedConvolutionLayer(lastLayerGUID, Vector3D<S32>(3,3,1), 16, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1), Vector3D<S32>(0,0,0));
		lastLayerGUID = pNetworkMaker->AddActivationLayer(lastLayerGUID, L"ReLU");

		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, 10, L"softmax_ALL_crossEntropy");
//		lastLayerGUID = pNetworkMaker->AddProbabilityArray2ValueLayer(lastLayerGUID, -1.0, 1.0f, 0.1f);
		lastLayerGUID = pNetworkMaker->AddSignalArray2ValueLayer(lastLayerGUID, -1.0f, 1.0f);

		// 出力レイヤー設定
		pNeuralNetwork->SetOutputLayerGUID(lastLayerGUID);
	}

	// 出力データ構造が正しいことを確認
	if(pNeuralNetwork->GetOutputDataStruct(&i_inputDataStruct, 1) != i_outputDataStruct)
	{
		layerDataManager.EraseLayerByGUID(pNeuralNetwork->GetGUID());
		return NULL;
	}


	// オプティマイザーの設定
	pNeuralNetwork->ChangeOptimizer(L"Adam");

	delete pNetworkMaker;

	return pNeuralNetwork;
}

Layer::Connect::ILayerConnectData* CreateNeuralNetwork(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct& inputDataStruct, const IODataStruct& outputDataStruct)
{
	return CreateNeuralNetwork_ver04(layerDLLManager, layerDataManager, inputDataStruct, outputDataStruct);
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

	// 実行時設定
	pNeuralNetworkLearn->SetRuntimeParameter(L"UseDropOut", true);
	pNeuralNetworkSample->SetRuntimeParameter(L"UseDropOut", false);
	
	pNeuralNetworkLearn->SetRuntimeParameter(L"GaussianNoise_Bias", 0.0f);
	pNeuralNetworkLearn->SetRuntimeParameter(L"GaussianNoise_Power", 0.0f);
	pNeuralNetworkSample->SetRuntimeParameter(L"GaussianNoise_Bias", 0.0f);
	pNeuralNetworkSample->SetRuntimeParameter(L"GaussianNoise_Power", 0.0f);


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


	std::vector<F32> lpOutputBuffer(pTeachOutputLayer->GetBufferCount() * BATCH_SIZE);
	std::vector<F32> lpTeachBuffer(pTeachOutputLayer->GetBufferCount() * BATCH_SIZE);

	// LSUV ( LAYER-SEQUENTIAL UNIT-VARIANCE INITIALIZATION ) を実行する
/*	{
		pNeuralNetworkLearn->SetRuntimeParameter(L"UpdateWeigthWithOutputVariance", true);
		pTeachInputLayer->PreProcessLoop();
		pNeuralNetworkLearn->PreProcessLoop();

		pTeachInputLayer->SetBatchDataNoList(pBatchDataNoListGenerator->GetBatchDataNoListByNum(0));

		pNeuralNetworkLearn->Calculate(pTeachInputLayer->GetOutputBuffer());

		pNeuralNetworkLearn->SetRuntimeParameter(L"UpdateWeigthWithOutputVariance", false);
	}*/


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
			pTeachInputLayer->PreProcessLoop();
			pTeachOutputLayer->PreProcessLoop();
			pNeuralNetworkLearn->PreProcessLoop();

			// 学習処理
			// バッチ単位で処理
			for(U32 batchNum=0; batchNum<pBatchDataNoListGenerator->GetBatchDataNoListCount(); batchNum++)
			{
#if USE_GPU
				if(batchNum%10 == 0)
#endif
				{
					printf(" L=%5.1f%%", (F32)batchNum * 100 / pBatchDataNoListGenerator->GetBatchDataNoListCount());
					printf("\b\b\b\b\b\b\b\b\b");
				}

				// データ切り替え
				pTeachInputLayer->SetBatchDataNoList(pBatchDataNoListGenerator->GetBatchDataNoListByNum(batchNum));
				pTeachOutputLayer->SetBatchDataNoList(pBatchDataNoListGenerator->GetBatchDataNoListByNum(batchNum));

				// 演算
				CONST_BATCH_BUFFER_POINTER lpInputBuffer[] = {pTeachInputLayer->GetOutputBuffer()};
				pNeuralNetworkLearn->Calculate(lpInputBuffer);

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
					U32 correctNo = (U32)((lpTeachBuffer[batchDataNum]  + 1.0)/2 * 10 + 0.5);
					U32 outputNo  = (U32)((lpOutputBuffer[batchDataNum] + 1.0)/2 * 10 + 0.5);

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
			pSampleInputLayer->PreProcessLoop();
			pSampleOutputLayer->PreProcessLoop();
			pNeuralNetworkSample->PreProcessLoop();

			// バッチ単位で処理
			for(U32 dataNum=0; dataNum<pSampleInputLayer->GetDataCount(); dataNum++)
			{
#if USE_GPU
				if(dataNum%10 == 0)
#endif
				{
					printf(" T=%5.1f%%", (F32)dataNum * 100 / pSampleInputLayer->GetDataCount());
					printf("\b\b\b\b\b\b\b\b\b");
				}

				// データ切り替え
				pSampleInputLayer->SetBatchDataNoList(&dataNum);
				pSampleOutputLayer->SetBatchDataNoList(&dataNum);

				// 演算
				CONST_BATCH_BUFFER_POINTER lpInputBuffer[] = {pSampleInputLayer->GetOutputBuffer()};
				pNeuralNetworkSample->Calculate(lpInputBuffer);

				// 誤差計算
				pSampleOutputLayer->CalculateLearnError(pNeuralNetworkSample->GetOutputBuffer());


				// 正解の番号を取得
				pSampleOutputLayer->GetOutputBuffer(&lpTeachBuffer[0]);
				pNeuralNetworkSample->GetOutputBuffer(&lpOutputBuffer[0]);
				{
					U32 correctNo = (U32)((lpTeachBuffer[0]  + 1.0)/2 * 10 + 0.5);
					U32 outputNo  = (U32)((lpOutputBuffer[0] + 1.0)/2 * 10 + 0.5);

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
	delete pBatchDataNoListGenerator;

	return ErrorCode::ERROR_CODE_NONE;
}