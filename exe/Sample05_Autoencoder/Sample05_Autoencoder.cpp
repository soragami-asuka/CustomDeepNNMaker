//=============================================
// オートエンコーダを実装するためのサンプル
//=============================================

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
#include"Utility/NeuralNetworkMaker.h"

#include"Library/NeuralNetwork/Initializer.h"

using namespace Gravisbell;

#define USE_GPU	1
#define USE_HOST_MEMORY 1

#define USE_BATCH_SIZE	128
#define MAX_EPOCH		20


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


/** ニューラルネットワーククラスを作成する */
Layer::Connect::ILayerConnectData* CreateEncoder0(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct& inputDataStruct, const IODataStruct& outputDataStruct);
Layer::Connect::ILayerConnectData* CreateDecoder0(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct& inputDataStruct, const IODataStruct& outputDataStruct);

Layer::Connect::ILayerConnectData* CreateAutoencoder0(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct& inputDataStruct, Layer::Connect::ILayerConnectData* pNNData_encoder0, Layer::Connect::ILayerConnectData* pNNData_decoder0);


/** ニューラルネットワークの学習とサンプル実行を同時実行 */
Gravisbell::ErrorCode LearnWithCalculateSampleError(
	Layer::NeuralNetwork::INeuralNetwork* pNeuralNetworkLearn,
	Layer::NeuralNetwork::INeuralNetwork* pNeuralNetworkSample,
	Layer::IOData::IIODataLayer* pTeachInputLayer,
	Layer::IOData::IIODataLayer* pSampleInputLayer,
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


	// レイヤーDLL管理クラスを作成
#if USE_GPU
	Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager = Gravisbell::Utility::NeuralNetworkLayer::CreateLayerDLLManagerGPU(L"./");
#else
	Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager = Gravisbell::Utility::NeuralNetworkLayer::CreateLayerDLLManagerCPU(L"./");
#endif
	if(pLayerDLLManager == NULL)
	{
		delete pDataLayerTeach_Input;
		delete pDataLayerTest_Input;
		delete pDataLayerTest_Output;
		return -1;
	}

	// レイヤーデータ管理クラスを作成
	Gravisbell::Layer::NeuralNetwork::ILayerDataManager* pLayerDataManager = Gravisbell::Layer::NeuralNetwork::CreateLayerDataManager();
	if(pLayerDataManager == NULL)
	{
		delete pDataLayerTeach_Input;
		delete pDataLayerTest_Input;
		delete pDataLayerTest_Output;
		delete pLayerDLLManager;
		return -1;
	}

	// 乱数を固定
#ifdef _DEBUG
//	Gravisbell::Layer::NeuralNetwork::GetInitializerManager().InitializeRandomParameter(0);
#endif

	// エンコーダ0を作成
	Gravisbell::Layer::Connect::ILayerConnectData* pNNData_encoder0 = CreateEncoder0(*pLayerDLLManager, *pLayerDataManager, pDataLayerTeach_Input->GetInputDataStruct(), 32);
	if(pNNData_encoder0 == NULL)
	{
		delete pDataLayerTeach_Input;
		delete pDataLayerTest_Input;
		delete pDataLayerTest_Output;
		delete pLayerDataManager;
		delete pLayerDLLManager;
		return -1;
	}

	// デコーダー0を作成
	Gravisbell::Layer::Connect::ILayerConnectData* pNNData_decoder0 = CreateDecoder0(*pLayerDLLManager, *pLayerDataManager, 32, pDataLayerTeach_Input->GetDataStruct());
	if(pNNData_decoder0 == NULL)
	{
		delete pDataLayerTeach_Input;
		delete pDataLayerTest_Input;
		delete pDataLayerTest_Output;
		delete pLayerDataManager;
		delete pLayerDLLManager;
		return -1;
	}

	// オートエンコーダを作成
	Gravisbell::Layer::Connect::ILayerConnectData* pNNData_autoencoder0 = CreateAutoencoder0(*pLayerDLLManager, *pLayerDataManager, pDataLayerTeach_Input->GetInputDataStruct(), pNNData_encoder0, pNNData_decoder0);
	if(pNNData_autoencoder0 == NULL)
	{
		delete pDataLayerTeach_Input;
		delete pDataLayerTest_Input;
		delete pDataLayerTest_Output;
		delete pLayerDataManager;
		delete pLayerDLLManager;
		return -1;
	}



	// 学習用ニューラルネットワーク作成
	Layer::NeuralNetwork::INeuralNetwork* pAutoencoder_learn = NULL;
	{
#if USE_HOST_MEMORY
		Layer::ILayerBase* pLayer = pNNData_autoencoder0->CreateLayer(boost::uuids::random_generator()().data, &pDataLayerTeach_Input->GetOutputDataStruct(), 1);
#else
		Layer::ILayerBase* pLayer = pNeuralNetworkData->CreateLayer_device(boost::uuids::random_generator()().data, &pDataLayerTeach_Input->GetOutputDataStruct(), 1);
#endif
		pAutoencoder_learn = dynamic_cast<Layer::NeuralNetwork::INeuralNetwork*>(pLayer);
		if(pAutoencoder_learn == NULL)
		{
			if(pLayer)
				delete pLayer;
		}
	}
	if(pAutoencoder_learn == NULL)
	{
		delete pDataLayerTeach_Input;
		delete pDataLayerTest_Input;
		delete pDataLayerTest_Output;
		delete pLayerDataManager;
		delete pLayerDLLManager;
		return -1;
	}

	// テスト用ニューラルネットワーク作成
	Layer::NeuralNetwork::INeuralNetwork* pAutoencoder_test = NULL;
	{
#if USE_HOST_MEMORY
		Layer::ILayerBase* pLayer = pNNData_autoencoder0->CreateLayer(boost::uuids::random_generator()().data, &pDataLayerTeach_Input->GetOutputDataStruct(), 1);
#else
		Layer::ILayerBase* pLayer = pNeuralNetworkData->CreateLayer_device(boost::uuids::random_generator()().data, &pDataLayerTeach_Input->GetOutputDataStruct(), 1);
#endif
		pAutoencoder_test = dynamic_cast<Layer::NeuralNetwork::INeuralNetwork*>(pLayer);
		if(pAutoencoder_test == NULL)
		{
			if(pLayer)
				delete pLayer;
		}
	}
	if(pAutoencoder_test == NULL)
	{
		delete pAutoencoder_learn;
		delete pDataLayerTeach_Input;
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
		if(::LearnWithCalculateSampleError(pAutoencoder_learn, pAutoencoder_test, pDataLayerTeach_Input, pDataLayerTest_Input, USE_BATCH_SIZE, MAX_EPOCH) != ErrorCode::ERROR_CODE_NONE)
		{
			delete pAutoencoder_learn;
			delete pAutoencoder_test;
			delete pDataLayerTeach_Input;
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
	delete pAutoencoder_learn;
	delete pAutoencoder_test;
	delete pDataLayerTeach_Input;
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


	*o_ppDataLayerTeach = Gravisbell::Layer::IOData::CreateIODataLayerCPU(dataStruct);
	*o_ppDataLayerTest  = Gravisbell::Layer::IOData::CreateIODataLayerCPU(dataStruct);


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



Layer::Connect::ILayerConnectData* CreateEncoder0(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
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

		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, 1024, L"ReLU");
		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, 512, L"ReLU");
		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, 256, L"ReLU");
		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, 128, L"ReLU");
		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, i_outputDataStruct.GetDataCount(), L"ReLU");

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
Layer::Connect::ILayerConnectData* CreateDecoder0(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
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
		
		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, i_inputDataStruct.GetDataCount(), L"ReLU");
		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, 128, L"ReLU");
		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, 256, L"ReLU");
		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, 512, L"ReLU");
		lastLayerGUID = pNetworkMaker->AddNeuralNetworkLayer_FA(lastLayerGUID, i_outputDataStruct.GetDataCount(), L"lenear");
		lastLayerGUID = pNetworkMaker->AddReshapeLayer(lastLayerGUID, i_outputDataStruct);

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

Layer::Connect::ILayerConnectData* CreateAutoencoder0(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct& i_inputDataStruct, Layer::Connect::ILayerConnectData* pNNData_encoder0, Layer::Connect::ILayerConnectData* pNNData_decoder0)
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

		lastLayerGUID = pNetworkMaker->AddLayer(lastLayerGUID, pNNData_encoder0, false);
		lastLayerGUID = pNetworkMaker->AddLayer(lastLayerGUID, pNNData_decoder0, false);

		// 出力レイヤー設定
		pNeuralNetwork->SetOutputLayerGUID(lastLayerGUID);
	}

	// 出力データ構造が正しいことを確認
	if(pNeuralNetwork->GetOutputDataStruct(&i_inputDataStruct, 1) != i_inputDataStruct)
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
	Layer::IOData::IIODataLayer* pSampleInputLayer,
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

	err = pNeuralNetworkSample->PreProcessCalculate(1);
	if(err != ErrorCode::ERROR_CODE_NONE)
		return err;
	err = pSampleInputLayer->PreProcessCalculate(1);
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


	// 学習を実行
	for(U32 learnTime=0; learnTime<LEARN_TIMES; learnTime++)
	{
		printf("%5d,", learnTime);

		// 学習
		{
			// 学習ループ先頭処理
//			pBatchDataNoListGenerator->PreProcessLearnLoop();
			pTeachInputLayer->PreProcessLoop();
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

				// 演算
				CONST_BATCH_BUFFER_POINTER lpInputBuffer[] = {pTeachInputLayer->GetOutputBuffer()};
				pNeuralNetworkLearn->Calculate(lpInputBuffer);

				// 誤差計算
				// 教師信号との誤差計算
				pTeachInputLayer->CalculateLearnError(pNeuralNetworkLearn->GetOutputBuffer());

				// 学習
				pNeuralNetworkLearn->Training(NULL, pTeachInputLayer->GetDInputBuffer());
			}
		}


		// サンプル実行
		{		
			// サンプル実行先頭処理
			pSampleInputLayer->PreProcessLoop();
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

				// 演算
				CONST_BATCH_BUFFER_POINTER lpInputBuffer[] = {pSampleInputLayer->GetOutputBuffer()};
				pNeuralNetworkSample->Calculate(lpInputBuffer);

				// 誤差計算
				pSampleInputLayer->CalculateLearnError(pNeuralNetworkSample->GetOutputBuffer());

			}
		}

		// 誤差表示
		{
			F32 errorMax, errorAve, errorAve2, errorCrossEntoropy;
			pTeachInputLayer->GetCalculateErrorValue(errorMax, errorAve, errorAve2, errorCrossEntoropy);
//			printf("学習：max=%.3f, ave=%.3f, ave2=%.3f, entropy=%.3f", errorMax, errorAve, errorAve2, errorCrossEntoropy);
			printf("%.3f,%.3f,%.3f - ",  errorMax, errorAve2, errorCrossEntoropy); 
		}
//		printf(" : ");
		{
			F32 errorMax, errorAve, errorAve2, errorCrossEntoropy;
			pSampleInputLayer->GetCalculateErrorValue(errorMax, errorAve, errorAve2, errorCrossEntoropy);
//			printf("実行：max=%.3f, ave=%.3f, ave2=%.3f, entropy=%.3f", errorMax, errorAve, errorAve2, errorCrossEntoropy);
			printf("%.3f,%.3f,%.3f", errorMax, errorAve2, errorCrossEntoropy); 
		}
		printf("\n");
	}

	// メモリ開放
	delete pBatchDataNoListGenerator;

	return ErrorCode::ERROR_CODE_NONE;
}