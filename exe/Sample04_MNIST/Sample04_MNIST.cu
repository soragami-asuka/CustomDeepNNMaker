// Sample04_MNIST.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
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

#define USE_BATCH_SIZE	128
#define MAX_EPOCH		5


/** データファイルをを読み込む
	@param	o_ppDataLayerTeach	教師データを格納したデータクラスの格納先ポインタアドレス
	@param	o_ppDataLayerTest	テストデータを格納したデータクラスの格納先ポインタアドレス
	@param	i_testRate			テストデータを全体の何%にするか0～1の間で設定
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
Layer::Connect::ILayerConnectData* CreateNeuralNetwork_ver02(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);
Layer::Connect::ILayerConnectData* CreateNeuralNetwork_ver03(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);
Layer::Connect::ILayerConnectData* CreateNeuralNetwork_ver04(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);
Layer::Connect::ILayerConnectData* CreateNeuralNetwork_ver05(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);
Layer::Connect::ILayerConnectData* CreateNeuralNetwork_ver06(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct);

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
//	Gravisbell::Layer::NeuralNetwork::GetInitializerManager().InitializeRandomParameter(0);
//#endif

	// ニューラルネットワーク作成
	Gravisbell::Layer::Connect::ILayerConnectData* pNeuralNetworkData = CreateNeuralNetwork_ver04(*pLayerDLLManager, *pLayerDataManager, pDataLayerTeach_Input->GetInputDataStruct(), pDataLayerTeach_Output->GetDataStruct());
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
	@param	i_testRate			テストデータを全体の何%にするか0～1の間で設定
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
	Gravisbell::IODataStruct dataStruct(10, 1, 1, 1);


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


	std::vector<F32> lpTmpBuf(dataStruct.ch);

	// データの見込み
	U32 dataCount = (U32)pDataFormat->GetVariableValue(L"images");
	U32 teachDataCount = (U32)(dataCount*(1.0f - i_testRate));
	for(U32 imageNum=0; imageNum<dataCount; imageNum++)
	{
		// U08 -> F32 変換
		for(U32 i=0; i<lpTmpBuf.size(); i++)
		{
			if(i == lpBuf[bufPos])
				lpTmpBuf[i] = 1.0f;
			else
				lpTmpBuf[i] = 0.0f;
		}

		if(imageNum < teachDataCount)
			(*o_ppDataLayerTeach)->AddData(&lpTmpBuf[0]);
		else
			(*o_ppDataLayerTest)->AddData(&lpTmpBuf[0]);

		bufPos += 1;
	}

	// データフォーマット削除
	delete pDataFormat;

	return Gravisbell::ErrorCode::ERROR_CODE_NONE;
}


/** ニューラルネットワーククラスを作成する */
Layer::Connect::ILayerConnectData* CreateNeuralNetwork(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
{
	using namespace Gravisbell::Utility::NeuralNetworkLayer;

	Gravisbell::ErrorCode err;

	// ニューラルネットワークを作成
	Layer::Connect::ILayerConnectData* pNeuralNetwork = CreateNeuralNetwork(layerDLLManager, layerDataManager);
	if(pNeuralNetwork == NULL)
		return NULL;


	// レイヤーを追加する
	if(pNeuralNetwork)
	{
		// 入力信号を直前レイヤーに設定
		Gravisbell::GUID lastLayerGUID = pNeuralNetwork->GetInputGUID();

		// ノイズレイヤー
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateGaussianNoiseLayer(layerDLLManager, layerDataManager, 0.0f, 0.1f), false);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;



		// 1層目
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateConvolutionLayer(layerDLLManager, layerDataManager,
			pNeuralNetwork->GetOutputDataStruct(lastLayerGUID, &i_inputDataStruct, 1).ch, Vector3D<S32>(5,5,1), 4, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0)), false);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreatePoolingLayer(layerDLLManager, layerDataManager, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1)), false);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#if USE_BATCHNORM
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateBatchNormalizationLayer(layerDLLManager, layerDataManager, pNeuralNetwork->GetOutputDataStruct(lastLayerGUID, &i_inputDataStruct, 1).ch), false);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#endif
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, L"ReLU"), false);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#if USE_DROPOUT
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateDropoutLayer(layerDLLManager, layerDataManager, 0.2f), false);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#endif


#if 1	// Single
		// 2層目
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateConvolutionLayer(layerDLLManager, layerDataManager, pNeuralNetwork->GetOutputDataStruct(lastLayerGUID, &i_inputDataStruct, 1).ch, Vector3D<S32>(5,5,1), 16, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0)), false);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreatePoolingLayer(layerDLLManager, layerDataManager, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1)), false);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#if USE_BATCHNORM
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateBatchNormalizationLayer(layerDLLManager, layerDataManager, pNeuralNetwork->GetOutputDataStruct(lastLayerGUID, &i_inputDataStruct, 1).ch), false);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#endif
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, L"ReLU"), false);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#if USE_DROPOUT
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateDropoutLayer(layerDLLManager, layerDataManager, 0.5f), false);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#endif

#if 0	// Expand
		// 3層目
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateConvolutionLayer(layerDLLManager, layerDataManager, pNeuralNetwork->GetOutputDataStruct(lastLayerGUID, &i_inputDataStruct, 1).ch, Vector3D<S32>(5,5,1), 16, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0)));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#if USE_BATCHNORM
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateBatchNormalizationLayer(layerDLLManager, layerDataManager, pNeuralNetwork->GetOutputDataStruct(lastLayerGUID, &i_inputDataStruct, 1).ch));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#endif
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, L"ReLU"));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#if USE_DROPOUT
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateDropoutLayer(layerDLLManager, layerDataManager, inputDataStruct, 0.5f));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#endif

		// 4層目
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateConvolutionLayer(layerDLLManager, layerDataManager, pNeuralNetwork->GetOutputDataStruct(lastLayerGUID, &i_inputDataStruct, 1).ch, Vector3D<S32>(5,5,1), 32, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0)));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreatePoolingLayer(layerDLLManager, layerDataManager, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1)));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#if USE_BATCHNORM
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateBatchNormalizationLayer(layerDLLManager, layerDataManager, pNeuralNetwork->GetOutputDataStruct(lastLayerGUID, &i_inputDataStruct, 1).ch));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#endif
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, L"ReLU"));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#if USE_DROPOUT
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateDropoutLayer(layerDLLManager, layerDataManager, inputDataStruct, 0.5f));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#endif

		// 5層目
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateConvolutionLayer(layerDLLManager, layerDataManager, pNeuralNetwork->GetOutputDataStruct(lastLayerGUID, &i_inputDataStruct, 1).ch, Vector3D<S32>(5,5,1), 32, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0)));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#if USE_BATCHNORM
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateBatchNormalizationLayer(layerDLLManager, layerDataManager, pNeuralNetwork->GetOutputDataStruct(lastLayerGUID, &i_inputDataStruct, 1).ch));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#endif
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, L"ReLU"));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#if USE_DROPOUT
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateDropoutLayer(layerDLLManager, layerDataManager, inputDataStruct, 0.5f));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#endif
#endif	// Expand

#elif 0	// MergeInput
		// 1層目のGUIDを記録
		Gravisbell::GUID lastLayerGUID_A = lastLayerGUID;
		Gravisbell::GUID lastLayerGUID_B = lastLayerGUID;

		// 2層目A
		{
			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				lastLayerGUID_A,
				CreateConvolutionLayer(layerDLLManager, layerDataManager, pNeuralNetwork->GetOutputDataStruct(lastLayerGUID_A, &i_inputDataStruct, 1).ch, Vector3D<S32>(5,5,1), 8, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0)));
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				lastLayerGUID_A,
				CreatePoolingLayer(layerDLLManager, layerDataManager, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1)));
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				lastLayerGUID_A,
				CreateBatchNormalizationLayer(layerDLLManager, layerDataManager, pNeuralNetwork->GetOutputDataStruct(lastLayerGUID_A, &i_inputDataStruct, 1).ch));
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				lastLayerGUID_A,
				CreateActivationLayer(layerDLLManager, layerDataManager, L"ReLU"));
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				lastLayerGUID_A,
				CreateDropoutLayer(layerDLLManager, layerDataManager, 0.5f));
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		}

		// 2層目B
		{
			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				lastLayerGUID_B,
				CreateConvolutionLayer(layerDLLManager, layerDataManager, pNeuralNetwork->GetOutputDataStruct(lastLayerGUID_B, &i_inputDataStruct, 1).ch, Vector3D<S32>(7,7,1), 8, Vector3D<S32>(1,1,1), Vector3D<S32>(3,3,0)));
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				lastLayerGUID_B,
				CreatePoolingLayer(layerDLLManager, layerDataManager, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1)));
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				lastLayerGUID_B,
				CreateBatchNormalizationLayer(layerDLLManager, layerDataManager, pNeuralNetwork->GetOutputDataStruct(lastLayerGUID_B, &i_inputDataStruct, 1).ch));
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				lastLayerGUID_B,
				CreateActivationLayer(layerDLLManager, layerDataManager, L"ReLU"));
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				lastLayerGUID_B,
				CreateDropoutLayer(layerDLLManager, layerDataManager, 0.5f));
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		}

		// A,B結合層
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateMergeInputLayer(layerDLLManager, layerDataManager),
			lastLayerGUID_A, lastLayerGUID_B);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#elif 1	// ResNet

		// ショートカットレイヤーを保存する
		Gravisbell::GUID lastLayerGUID_shortCut = lastLayerGUID;

		// 2層目
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateConvolutionLayer(layerDLLManager, layerDataManager, pNeuralNetwork->GetOutputDataStruct(lastLayerGUID, &i_inputDataStruct, 1).ch, Vector3D<S32>(5,5,1), 8, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0)));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, L"ReLU"));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;

		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateConvolutionLayer(layerDLLManager, layerDataManager, pNeuralNetwork->GetOutputDataStruct(lastLayerGUID, &i_inputDataStruct, 1).ch, Vector3D<S32>(5,5,1), 8, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0)));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, L"ReLU"));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;

		// 残差レイヤー
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateResidualLayer(layerDLLManager, layerDataManager),
			lastLayerGUID, lastLayerGUID_shortCut);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;

		//// A,B結合層
		//err = AddLayerToNetworkLast(
		//	*pNeuralNetwork,
		//	lppLayerData,
		//	inputDataStruct, lastLayerGUID,
		//	CreateMergeInputLayer(layerDLLManager, inputDataStruct, inputDataStruct_shortCut),
		//	lastLayerGUID, lastLayerGUID_shortCut);
		//if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;

		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreatePoolingLayer(layerDLLManager, layerDataManager, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1)));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		//err = AddLayerToNetworkLast(
		//	*pNeuralNetwork,
		//	lppLayerData,
		//	inputDataStruct, lastLayerGUID,
		//	CreateBatchNormalizationLayer(layerDLLManager, inputDataStruct));
		//if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, L"ReLU"));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		//err = AddLayerToNetworkLast(
		//	*pNeuralNetwork,
		//	lppLayerData,
		//	inputDataStruct, lastLayerGUID,
		//	CreateDropoutLayer(layerDLLManager, inputDataStruct, 0.5f));
		//if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#elif 0// UpSampling

		// 2層目
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateUpSamplingLayer(layerDLLManager, layerDataManager, Vector3D<S32>(2,2,1), true));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;

		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateConvolutionLayer(layerDLLManager, layerDataManager, pNeuralNetwork->GetOutputDataStruct(lastLayerGUID, &i_inputDataStruct, 1).ch, Vector3D<S32>(5,5,1), 8, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0)));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreatePoolingLayer(layerDLLManager, layerDataManager, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1)));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		//err = AddLayerToNetworkLast(
		//	*pNeuralNetwork,
		//	lppLayerData,
		//	inputDataStruct, lastLayerGUID,
		//	CreateBatchNormalizationLayer(layerDLLManager, inputDataStruct));
		//if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, L"ReLU"));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		//err = AddLayerToNetworkLast(
		//	*pNeuralNetwork,
		//	lppLayerData,
		//	inputDataStruct, lastLayerGUID,
		//	CreateDropoutLayer(layerDLLManager, inputDataStruct, 0.5f));
		//if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#else
#endif


		// 3層目
#if 1	// 全結合
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateFullyConnectLayer(layerDLLManager, layerDataManager, pNeuralNetwork->GetOutputDataStruct(lastLayerGUID, &i_inputDataStruct, 1).GetDataCount(), i_outputDataStruct.GetDataCount()), false);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, L"softmax_ALL_crossEntropy"), false);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#elif 0	// GlobalAveragePooling
		// 畳み込み(出力：2ch)
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateConvolutionLayer(layerDLLManager, layerDataManager, inputDataStruct, Vector3D<S32>(5,5,1), outputDataStruct.GetDataCount(), Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0)));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		// Pooling
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateGlobalAveragePoolingLayer(layerDLLManager, layerDataManager, inputDataStruct));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		// 活性化
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, inputDataStruct, L"softmax_ALL_crossEntropy"));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#else
#endif

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
	pNeuralNetwork->ChangeOptimizer(L"SGD");
	pNeuralNetwork->SetOptimizerHyperParameter(L"LearnCoeff", 0.005f);
//	pNeuralNetwork->ChangeOptimizer(L"Adam");

	return pNeuralNetwork;
}

/** ニューラルネットワーククラスを作成する */
Layer::Connect::ILayerConnectData* CreateNeuralNetwork_ver02(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
{
	using namespace Gravisbell::Utility::NeuralNetworkLayer;

	Gravisbell::ErrorCode err;

	// ニューラルネットワークを作成
	Layer::Connect::ILayerConnectData* pNeuralNetwork = CreateNeuralNetwork(layerDLLManager, layerDataManager);
	if(pNeuralNetwork == NULL)
		return NULL;


	// レイヤーを追加する
	if(pNeuralNetwork)
	{
		// 入力信号を直前レイヤーに設定
		Gravisbell::GUID lastLayerGUID = pNeuralNetwork->GetInputGUID();

		// 1層目
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateConvolutionLayer(layerDLLManager, layerDataManager,
			pNeuralNetwork->GetOutputDataStruct(lastLayerGUID, &i_inputDataStruct, 1).ch, Vector3D<S32>(5,5,1), 4, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0)), false);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreatePoolingLayer(layerDLLManager, layerDataManager, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1)), false);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, L"ReLU"), false);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;

		// 正規化レイヤー
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateBatchNormalizationAllLayer(layerDLLManager, layerDataManager), false);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;

		// 2層目
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateConvolutionLayer(layerDLLManager, layerDataManager,
			pNeuralNetwork->GetOutputDataStruct(lastLayerGUID, &i_inputDataStruct, 1).ch, Vector3D<S32>(5,5,1), 16, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0)), false);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreatePoolingLayer(layerDLLManager, layerDataManager, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1)), false);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, L"ReLU"), false);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;


		// チャンネル分割
		Gravisbell::GUID lastLayerGUID_chA = lastLayerGUID;
		Gravisbell::GUID lastLayerGUID_chB = lastLayerGUID;
		Gravisbell::GUID lastLayerGUID_chC = lastLayerGUID;

		// A
		{
			// 分割
			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				lastLayerGUID_chA,
				CreateChooseChannelLayer(layerDLLManager, layerDataManager, 0, 4), false );
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;


			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				lastLayerGUID_chA,
				CreateConvolutionLayer(layerDLLManager, layerDataManager,
				pNeuralNetwork->GetOutputDataStruct(lastLayerGUID_chA, &i_inputDataStruct, 1).ch, Vector3D<S32>(5,5,1), 8, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0)), false);
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				lastLayerGUID_chA,
				CreateActivationLayer(layerDLLManager, layerDataManager, L"ReLU"), false);
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		}
		// B
		{
			// 分割
			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				lastLayerGUID_chB,
				CreateChooseChannelLayer(layerDLLManager, layerDataManager, 4, 4), false );
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;


			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				lastLayerGUID_chB,
				CreateConvolutionLayer(layerDLLManager, layerDataManager,
				pNeuralNetwork->GetOutputDataStruct(lastLayerGUID_chB, &i_inputDataStruct, 1).ch, Vector3D<S32>(5,5,1), 8, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0)), false);
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				lastLayerGUID_chB,
				CreateActivationLayer(layerDLLManager, layerDataManager, L"ReLU"), false);
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		}
		// C
		{
			// 分割
			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				lastLayerGUID_chC,
				CreateChooseChannelLayer(layerDLLManager, layerDataManager, 8, 8), false );
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;

			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				lastLayerGUID_chC,
				CreateConvolutionLayer(layerDLLManager, layerDataManager,
				pNeuralNetwork->GetOutputDataStruct(lastLayerGUID_chC, &i_inputDataStruct, 1).ch, Vector3D<S32>(5,5,1), 16, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0)), false);
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				lastLayerGUID_chC,
				CreateActivationLayer(layerDLLManager, layerDataManager, L"ReLU"), false);
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		}

		// マージ
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateMergeInputLayer(layerDLLManager, layerDataManager), false,
			lastLayerGUID_chA, lastLayerGUID_chB, lastLayerGUID_chC);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;

		// 4層目
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateConvolutionLayer(layerDLLManager, layerDataManager,
			pNeuralNetwork->GetOutputDataStruct(lastLayerGUID, &i_inputDataStruct, 1).ch, Vector3D<S32>(5,5,1), 32, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0)), false);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreatePoolingLayer(layerDLLManager, layerDataManager, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1)), false);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, L"ReLU"), false);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;


		// 全結合
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateFullyConnectLayer(layerDLLManager, layerDataManager, pNeuralNetwork->GetOutputDataStruct(lastLayerGUID, &i_inputDataStruct, 1).GetDataCount(), i_outputDataStruct.GetDataCount()), false);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, L"softmax_ALL_crossEntropy"), false);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;


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
//	pNeuralNetwork->ChangeOptimizer(L"SGD");
//	pNeuralNetwork->SetOptimizerHyperParameter(L"LearnCoeff", 0.005f);
	pNeuralNetwork->ChangeOptimizer(L"Adam");

	return pNeuralNetwork;
}

/** ニューラルネットワーククラスを作成する */
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
		Gravisbell::GUID lastLayerGUID = pNeuralNetwork->GetInputGUID();

		// 1層目
		lastLayerGUID = pNetworkMaker->AddConvolutionLayer(lastLayerGUID, Vector3D<S32>(5,5,1), 4, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0), L"he_normal");
		lastLayerGUID = pNetworkMaker->AddPoolingLayer(lastLayerGUID, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1));
		lastLayerGUID = pNetworkMaker->AddActivationLayer(lastLayerGUID, L"ReLU");

		// 正規化レイヤー
		lastLayerGUID = pNetworkMaker->AddNormalizationScaleLayer(lastLayerGUID);

		// 2層目
		lastLayerGUID = pNetworkMaker->AddConvolutionLayer(lastLayerGUID, Vector3D<S32>(5,5,1), 16, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0), L"he_normal");
		lastLayerGUID = pNetworkMaker->AddPoolingLayer(lastLayerGUID, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1));
		lastLayerGUID = pNetworkMaker->AddActivationLayer(lastLayerGUID, L"ReLU");

		// チャンネル分割
		Gravisbell::GUID lastLayerGUID_chA = lastLayerGUID;
		Gravisbell::GUID lastLayerGUID_chB = lastLayerGUID;
		Gravisbell::GUID lastLayerGUID_chC = lastLayerGUID;

		// A
		{
			// 分割
			lastLayerGUID_chA = pNetworkMaker->AddChooseChannelLayer(lastLayerGUID_chA, 0, 4);

			lastLayerGUID_chA = pNetworkMaker->AddConvolutionLayer(lastLayerGUID_chA, Vector3D<S32>(5,5,1), 8, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0), L"he_normal");
			lastLayerGUID_chA = pNetworkMaker->AddActivationLayer(lastLayerGUID_chA, L"ReLU");
		}
		// B
		{
			// 分割
			lastLayerGUID_chB = pNetworkMaker->AddChooseChannelLayer(lastLayerGUID_chB, 4, 4);

			lastLayerGUID_chB = pNetworkMaker->AddConvolutionLayer(lastLayerGUID_chB, Vector3D<S32>(5,5,1), 8, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0), L"he_normal");
			lastLayerGUID_chB = pNetworkMaker->AddActivationLayer(lastLayerGUID_chB, L"ReLU");
		}
		// C
		{
			// 分割
			lastLayerGUID_chC = pNetworkMaker->AddChooseChannelLayer(lastLayerGUID_chC, 8, 8);

			lastLayerGUID_chC = pNetworkMaker->AddConvolutionLayer(lastLayerGUID_chC, Vector3D<S32>(5,5,1), 16, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0), L"he_normal");
			lastLayerGUID_chC = pNetworkMaker->AddActivationLayer(lastLayerGUID_chC, L"ReLU");
		}

		// マージ
		lastLayerGUID = pNetworkMaker->AddMergeInputLayer(lastLayerGUID_chA, lastLayerGUID_chB, lastLayerGUID_chC);

		// 4層目
		lastLayerGUID = pNetworkMaker->AddConvolutionLayer(lastLayerGUID, Vector3D<S32>(5,5,1), 32, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0), L"he_normal");
		lastLayerGUID = pNetworkMaker->AddPoolingLayer(lastLayerGUID, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1));
		lastLayerGUID = pNetworkMaker->AddActivationLayer(lastLayerGUID, L"ReLU");

		// 全結合
		lastLayerGUID = pNetworkMaker->AddFullyConnectLayer(lastLayerGUID, i_outputDataStruct.GetDataCount(), L"glorot_normal");
		lastLayerGUID = pNetworkMaker->AddActivationLayer(lastLayerGUID, L"softmax_ALL_crossEntropy");


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
//	pNeuralNetwork->ChangeOptimizer(L"SGD");
//	pNeuralNetwork->SetOptimizerHyperParameter(L"LearnCoeff", 0.005f);
	pNeuralNetwork->ChangeOptimizer(L"Adam");

	delete pNetworkMaker;

	return pNeuralNetwork;
}

/** ニューラルネットワーククラスを作成する */
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
		Gravisbell::GUID lastLayerGUID = pNeuralNetwork->GetInputGUID();

//		lastLayerGUID = pNetworkMaker->AddReshapeMirrorXLayer(lastLayerGUID);
		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_CAD(lastLayerGUID, Vector3D<S32>(5,5,1),  4, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0), L"ReLU", 0.5f);
		lastLayerGUID = pNetworkMaker->AddPoolingLayer(lastLayerGUID, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1));
		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_CAD(lastLayerGUID, Vector3D<S32>(5,5,1),  8, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0), L"ReLU", 0.5f);
		lastLayerGUID = pNetworkMaker->AddPoolingLayer(lastLayerGUID, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1));
		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_CAD(lastLayerGUID, Vector3D<S32>(5,5,1), 16, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0), L"ReLU", 0.5f);
		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_CBAD(lastLayerGUID, Vector3D<S32>(5,5,1), 32, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0), L"ReLU", 0.5f);

		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, 64, L"ReLU");
		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, 32, L"ReLU");
		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, i_outputDataStruct.GetDataCount(), L"softmax_ALL_crossEntropy");

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
		Gravisbell::GUID lastLayerGUID = pNeuralNetwork->GetInputGUID();

		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_CA(lastLayerGUID, Vector3D<S32>(3,3,1),  1, Vector3D<S32>(1,1,1), Vector3D<S32>(1,1,0), L"ReLU");
		lastLayerGUID = pNetworkMaker->AddReshapeLayer(lastLayerGUID, IODataStruct(14, 56, 1, 1));
		lastLayerGUID = pNetworkMaker->AddNormalizationScaleLayer(lastLayerGUID);
		lastLayerGUID = pNetworkMaker->AddReshapeSquareZeroSideLeftTopLayer(lastLayerGUID, 10, 6);
		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_CAD(lastLayerGUID, Vector3D<S32>(5,5,1),  4, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0), L"ReLU", 0.5f);
		lastLayerGUID = pNetworkMaker->AddPoolingLayer(lastLayerGUID, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1));
		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_CAD(lastLayerGUID, Vector3D<S32>(5,5,1),  8, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0), L"ReLU", 0.5f);
		lastLayerGUID = pNetworkMaker->AddPoolingLayer(lastLayerGUID, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1));
		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_CAD(lastLayerGUID, Vector3D<S32>(5,5,1), 16, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0), L"ReLU", 0.5f);
		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_CAD(lastLayerGUID, Vector3D<S32>(5,5,1), 32, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0), L"ReLU", 0.5f);

		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, 64, L"ReLU");
		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, 32, L"ReLU");
		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, i_outputDataStruct.GetDataCount(), L"softmax_ALL_crossEntropy");

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


Layer::Connect::ILayerConnectData* CreateNeuralNetwork_ver06(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
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
		Gravisbell::GUID lastLayerGUID = pNeuralNetwork->GetInputGUID();

		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, 1024, L"ReLU");
		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, 512, L"ReLU");
		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, 256, L"ReLU");
		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, 128, L"ReLU");
		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, 64, L"ReLU");
		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, 32, L"ReLU");
		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, i_outputDataStruct.GetDataCount(), L"softmax_ALL_crossEntropy");

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
	{
		pNeuralNetworkLearn->SetRuntimeParameter(L"UpdateWeigthWithOutputVariance", true);
		pTeachInputLayer->PreProcessLoop();
		pNeuralNetworkLearn->PreProcessLoop();

		pTeachInputLayer->SetBatchDataNoList(pBatchDataNoListGenerator->GetBatchDataNoListByNum(0));

		pNeuralNetworkLearn->Calculate(pTeachInputLayer->GetOutputBuffer());

		pNeuralNetworkLearn->SetRuntimeParameter(L"UpdateWeigthWithOutputVariance", false);
	}


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
	delete pBatchDataNoListGenerator;

	return ErrorCode::ERROR_CODE_NONE;
}