// Sample04_MNIST.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include<crtdbg.h>

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

using namespace Gravisbell;

#define USE_GPU	1
#define USE_HOST_MEMORY 1

#define USE_BATCHNORM	1
#define USE_DROPOUT		0


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


	// ニューラルネットワーク作成
	Gravisbell::Layer::ILayerData* pNeuralNetworkData = CreateNeuralNetwork(*pLayerDLLManager, *pLayerDataManager, pDataLayerTeach_Input->GetInputDataStruct(), pDataLayerTeach_Output->GetDataStruct());
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
	printf("バイナリファイル読み込み\n");
	Gravisbell::Layer::ILayerData* pNeuralNetworkData2 = NULL;
	Gravisbell::Utility::NeuralNetworkLayer::ReadNetworkFromBinaryFile(*pLayerDLLManager, &pNeuralNetworkData2,  L"../../LayerData/test.bin");
	// 別ファイルに保存する
	printf("バイナリファイル保存2\n");
	Gravisbell::Utility::NeuralNetworkLayer::WriteNetworkToBinaryFile(*pNeuralNetworkData2,  L"../../LayerData/test2.bin");
	printf("終了\n");
	delete pNeuralNetworkData2;

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
		Layer::ILayerBase* pLayer = pNeuralNetworkData->CreateLayer(boost::uuids::random_generator()().data);
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
		Layer::ILayerBase* pLayer = pNeuralNetworkData->CreateLayer(boost::uuids::random_generator()().data);
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
		// 学習
		if(::LearnWithCalculateSampleError(pNeuralNetworkLearn, pNeuralNetworkTest, pDataLayerTeach_Input, pDataLayerTeach_Output, pDataLayerTest_Input, pDataLayerTest_Output, 32, 20) != ErrorCode::ERROR_CODE_NONE)
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

	}


	// バッファ開放
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
	*o_ppDataLayerTeach = Gravisbell::Layer::IOData::CreateIODataLayerGPU_host(dataStruct);
	*o_ppDataLayerTest  = Gravisbell::Layer::IOData::CreateIODataLayerGPU_host(dataStruct);
#else
	*o_ppDataLayerTeach = Gravisbell::Layer::IOData::CreateIODataLayerGPU_device(dataStruct);
	*o_ppDataLayerTest  = Gravisbell::Layer::IOData::CreateIODataLayerGPU_device(dataStruct);
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
	*o_ppDataLayerTeach = Gravisbell::Layer::IOData::CreateIODataLayerGPU_host(dataStruct);
	*o_ppDataLayerTest  = Gravisbell::Layer::IOData::CreateIODataLayerGPU_host(dataStruct);
#else
	*o_ppDataLayerTeach = Gravisbell::Layer::IOData::CreateIODataLayerGPU_device(dataStruct);
	*o_ppDataLayerTest  = Gravisbell::Layer::IOData::CreateIODataLayerGPU_device(dataStruct);
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
Layer::Connect::ILayerConnectData* CreateNeuralNetwork(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct& inputDataStruct, const IODataStruct& outputDataStruct)
{
	using namespace Gravisbell::Utility::NeuralNetworkLayer;

	Gravisbell::ErrorCode err;

	// ニューラルネットワークを作成
	Layer::Connect::ILayerConnectData* pNeuralNetwork = CreateNeuralNetwork(layerDLLManager, layerDataManager, inputDataStruct);
	if(pNeuralNetwork == NULL)
		return NULL;


	// レイヤーを追加する
	if(Layer::IO::ISingleInputLayerData* pNeuralNetworkInput = dynamic_cast<Layer::IO::ISingleInputLayerData*>(pNeuralNetwork))
	{
		// 入力信号を直前レイヤーに設定
		Gravisbell::IODataStruct inputDataStruct = pNeuralNetworkInput->GetInputDataStruct();
		Gravisbell::GUID lastLayerGUID = pNeuralNetwork->GetInputGUID();

		// 1層目
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateConvolutionLayer(layerDLLManager, layerDataManager, inputDataStruct, Vector3D<S32>(5,5,1), 4, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0)));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreatePoolingLayer(layerDLLManager, layerDataManager, inputDataStruct, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1)));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateBatchNormalizationLayer(layerDLLManager, layerDataManager, inputDataStruct));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#if USE_BATCHNORM
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, inputDataStruct, L"ReLU"));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#endif
#if USE_DROPOUT
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateDropoutLayer(layerDLLManager, layerDataManager, inputDataStruct, 0.2f));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#endif


#if 0	// Single
		// 2層目
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateConvolutionLayer(layerDLLManager, layerDataManager, inputDataStruct, Vector3D<S32>(5,5,1), 16, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0)));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreatePoolingLayer(layerDLLManager, layerDataManager, inputDataStruct, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1)));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateBatchNormalizationLayer(layerDLLManager, layerDataManager, inputDataStruct));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#if USE_BATCHNORM
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, inputDataStruct, L"ReLU"));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#endif
#if USE_DROPOUT
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateDropoutLayer(layerDLLManager, layerDataManager, inputDataStruct, 0.5f));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#endif

#if 1	// Expand
		// 3層目
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateConvolutionLayer(layerDLLManager, layerDataManager, inputDataStruct, Vector3D<S32>(5,5,1), 16, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0)));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateBatchNormalizationLayer(layerDLLManager, layerDataManager, inputDataStruct));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#if USE_BATCHNORM
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, inputDataStruct, L"ReLU"));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#endif
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
			inputDataStruct, lastLayerGUID,
			CreateConvolutionLayer(layerDLLManager, layerDataManager, inputDataStruct, Vector3D<S32>(5,5,1), 32, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0)));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreatePoolingLayer(layerDLLManager, layerDataManager, inputDataStruct, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1)));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateBatchNormalizationLayer(layerDLLManager, layerDataManager, inputDataStruct));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#if USE_BATCHNORM
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, inputDataStruct, L"ReLU"));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#endif
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
			inputDataStruct, lastLayerGUID,
			CreateConvolutionLayer(layerDLLManager, layerDataManager, inputDataStruct, Vector3D<S32>(5,5,1), 32, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0)));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateBatchNormalizationLayer(layerDLLManager, layerDataManager, inputDataStruct));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#if USE_BATCHNORM
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, inputDataStruct, L"ReLU"));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#endif
#if USE_DROPOUT
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateDropoutLayer(layerDLLManager, layerDataManager, inputDataStruct, 0.5f));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#endif
#endif	// Expand

#elif 1	// MergeInput
		// 1層目のGUIDを記録
		Gravisbell::GUID lastLayerGUID_A = lastLayerGUID;
		Gravisbell::GUID lastLayerGUID_B = lastLayerGUID;
		Gravisbell::IODataStruct inputDataStruct_A = inputDataStruct;
		Gravisbell::IODataStruct inputDataStruct_B = inputDataStruct;

		// 2層目A
		{
			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				inputDataStruct_A, lastLayerGUID_A,
				CreateConvolutionLayer(layerDLLManager, layerDataManager, inputDataStruct_A, Vector3D<S32>(5,5,1), 8, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0)));
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				inputDataStruct_A, lastLayerGUID_A,
				CreatePoolingLayer(layerDLLManager, layerDataManager, inputDataStruct_A, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1)));
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				inputDataStruct_A, lastLayerGUID_A,
				CreateBatchNormalizationLayer(layerDLLManager, layerDataManager, inputDataStruct_A));
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				inputDataStruct_A, lastLayerGUID_A,
				CreateActivationLayer(layerDLLManager, layerDataManager, inputDataStruct_A, L"ReLU"));
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				inputDataStruct_A, lastLayerGUID_A,
				CreateDropoutLayer(layerDLLManager, layerDataManager, inputDataStruct_A, 0.5f));
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		}

		// 2層目B
		{
			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				inputDataStruct_B, lastLayerGUID_B,
				CreateConvolutionLayer(layerDLLManager, layerDataManager, inputDataStruct_B, Vector3D<S32>(7,7,1), 8, Vector3D<S32>(1,1,1), Vector3D<S32>(3,3,0)));
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				inputDataStruct_B, lastLayerGUID_B,
				CreatePoolingLayer(layerDLLManager, layerDataManager, inputDataStruct_B, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1)));
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				inputDataStruct_B, lastLayerGUID_B,
				CreateBatchNormalizationLayer(layerDLLManager, layerDataManager, inputDataStruct_B));
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				inputDataStruct_B, lastLayerGUID_B,
				CreateActivationLayer(layerDLLManager, layerDataManager, inputDataStruct_B, L"ReLU"));
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
			err = AddLayerToNetworkLast(
				*pNeuralNetwork,
				inputDataStruct_B, lastLayerGUID_B,
				CreateDropoutLayer(layerDLLManager, layerDataManager, inputDataStruct_B, 0.5f));
			if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		}

		// A,B結合層
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateMergeInputLayer(layerDLLManager, layerDataManager, inputDataStruct_A, inputDataStruct_B),
			lastLayerGUID_A, lastLayerGUID_B);
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
#elif 0	// ResNet

		// ショートカットレイヤーを保存する
		Gravisbell::GUID lastLayerGUID_shortCut = lastLayerGUID;
		Gravisbell::IODataStruct inputDataStruct_shortCut = inputDataStruct;

		// 2層目
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateConvolutionLayer(layerDLLManager, layerDataManager, inputDataStruct, Vector3D<S32>(5,5,1), 8, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0)));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, inputDataStruct, L"ReLU"));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;

		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateConvolutionLayer(layerDLLManager, layerDataManager, inputDataStruct, Vector3D<S32>(5,5,1), 8, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0)));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, inputDataStruct, L"ReLU"));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;

		// 残差レイヤー
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateResidualLayer(layerDLLManager, layerDataManager, inputDataStruct, inputDataStruct_shortCut),
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
			inputDataStruct, lastLayerGUID,
			CreatePoolingLayer(layerDLLManager, layerDataManager, inputDataStruct, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1)));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		//err = AddLayerToNetworkLast(
		//	*pNeuralNetwork,
		//	lppLayerData,
		//	inputDataStruct, lastLayerGUID,
		//	CreateBatchNormalizationLayer(layerDLLManager, inputDataStruct));
		//if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, inputDataStruct, L"ReLU"));
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
			inputDataStruct, lastLayerGUID,
			CreateUpSamplingLayer(layerDLLManager, layerDataManager, inputDataStruct, Vector3D<S32>(2,2,1), true));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;

		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateConvolutionLayer(layerDLLManager, layerDataManager, inputDataStruct, Vector3D<S32>(5,5,1), 8, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0)));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreatePoolingLayer(layerDLLManager, layerDataManager, inputDataStruct, Vector3D<S32>(2,2,1), Vector3D<S32>(2,2,1)));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		//err = AddLayerToNetworkLast(
		//	*pNeuralNetwork,
		//	lppLayerData,
		//	inputDataStruct, lastLayerGUID,
		//	CreateBatchNormalizationLayer(layerDLLManager, inputDataStruct));
		//if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, inputDataStruct, L"ReLU"));
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
			inputDataStruct, lastLayerGUID,
			CreateFullyConnectLayer(layerDLLManager, layerDataManager, inputDataStruct, outputDataStruct.GetDataCount()));
		if(err != ErrorCode::ERROR_CODE_NONE)	return NULL;
		err = AddLayerToNetworkLast(
			*pNeuralNetwork,
			inputDataStruct, lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, inputDataStruct, L"softmax_ALL_crossEntropy"));
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

	// オプティマイザーの設定
	pNeuralNetwork->ChangeOptimizer(L"SGD");
	pNeuralNetwork->SetOptimizerHyperParameter(L"LearnCoeff", 0.005f);
//	pNeuralNetwork->ChangeOptimizer(L"Adam");

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

	// 学習係数を設定
#if USE_BATCHNORM
	pNeuralNetworkLearn->SetLearnSettingData(L"LearnCoeff", 0.005f);
#else
	pNeuralNetworkLearn->SetLearnSettingData(L"LearnCoeff", 0.001f);
#endif
//	pNeuralNetworkLearn->SetLearnSettingData(L"Optimizer", L"SGD");
//	pNeuralNetworkLearn->SetLearnSettingData(L"Optimizer", L"Momentum");
//	pNeuralNetworkLearn->SetLearnSettingData(L"Optimizer", L"AdaDelta");
	pNeuralNetworkLearn->SetLearnSettingData(L"Optimizer", L"Adam");
	pNeuralNetworkLearn->SetLearnSettingData(L"Momentum_alpha", 0.9f);
	pNeuralNetworkLearn->SetLearnSettingData(L"AdaDelta_rho", (F32)0.95f);
	pNeuralNetworkLearn->SetLearnSettingData(L"AdaDelta_epsilon", (F32)1e-8);

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
			pSampleInputLayer->PreProcessCalculateLoop();
			pSampleOutputLayer->PreProcessCalculateLoop();
			pNeuralNetworkSample->PreProcessCalculateLoop();

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
	delete pLearnSetting;
	delete pBatchDataNoListGenerator;

	return ErrorCode::ERROR_CODE_NONE;
}