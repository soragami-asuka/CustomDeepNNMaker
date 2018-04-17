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
#include"Library/NeuralNetwork/Initializer.h"
#include"Library/Layer/IOData/IODataLayer.h"
#include"Layer/ILayerDataSOM.h"
#include"Layer/Connect/ILayerConnectData.h"
#include"Layer/NeuralNetwork/INeuralNetwork.h"
#include"Utility/NeuralNetworkLayer.h"
#include"Utility/NeuralNetworkMaker.h"

#include"opencv2/core/core.hpp"
#include"opencv2/highgui/highgui.hpp"

using namespace Gravisbell;

#define USE_GPU	0
#define USE_HOST_MEMORY 1

#define USE_BATCH_SIZE	1
#define MAX_EPOCH		10000

#define RESOLUTION_COUNT	(20)
#define IMAGE_WIDTH			(512)
#define RECT_WIDTH			(IMAGE_WIDTH / RESOLUTION_COUNT)


/** ニューラルネットワーククラスを作成する */
Layer::Connect::ILayerConnectData* CreateNeuralNetwork(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct& inputDataStruct, const IODataStruct& outputDataStruct, Gravisbell::Layer::ILayerDataSOM** o_ppLayerDataSOM);


/** ニューラルネットワークの学習とサンプル実行を同時実行 */
Gravisbell::ErrorCode LearnWithCalculateSampleError(
	Layer::NeuralNetwork::INeuralNetwork* pNeuralNetwork,
	Gravisbell::Layer::ILayerDataSOM& layerDataSOM,
	const U32 BATCH_SIZE,
	const U32 LEARN_TIMES);


int _tmain(int argc, _TCHAR* argv[])
{
#ifdef _DEBUG
	::_CrtSetDbgFlag(_CRTDBG_LEAK_CHECK_DF | _CRTDBG_ALLOC_MEM_DF);
#endif


	// レイヤーDLL管理クラスを作成
#if USE_GPU
	Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager = Gravisbell::Utility::NeuralNetworkLayer::CreateLayerDLLManagerGPU(L"./");
#else
	Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager = Gravisbell::Utility::NeuralNetworkLayer::CreateLayerDLLManagerCPU(L"./");
#endif
	if(pLayerDLLManager == NULL)
	{
		return -1;
	}

	// レイヤーデータ管理クラスを作成
	Gravisbell::Layer::NeuralNetwork::ILayerDataManager* pLayerDataManager = Gravisbell::Layer::NeuralNetwork::CreateLayerDataManager();
	if(pLayerDataManager == NULL)
	{
		delete pLayerDLLManager;
		return -1;
	}

	// 乱数を固定
#ifdef _DEBUG
//	Gravisbell::Layer::NeuralNetwork::GetInitializerManager().InitializeRandomParameter(0);
#endif

	// エンコーダ0を作成
	Gravisbell::Layer::ILayerDataSOM* pLayerDataSOM = NULL;
	Gravisbell::Layer::Connect::ILayerConnectData* pNNData = CreateNeuralNetwork(*pLayerDLLManager, *pLayerDataManager, 3, 2, &pLayerDataSOM);
	if(pNNData == NULL)
	{
		delete pLayerDataManager;
		delete pLayerDLLManager;
		return -1;
	}


	// 学習用ニューラルネットワーク作成
	Layer::NeuralNetwork::INeuralNetwork* pNeuralNetwork = NULL;
	{
		Layer::ILayerBase* pLayer = pNNData->CreateLayer(boost::uuids::random_generator()().data, &IODataStruct(3), 1);

		pNeuralNetwork = dynamic_cast<Layer::NeuralNetwork::INeuralNetwork*>(pLayer);
		if(pNeuralNetwork == NULL)
		{
			if(pLayer)
				delete pLayer;
		}
	}
	if(pNeuralNetwork == NULL)
	{
		delete pLayerDataManager;
		delete pLayerDLLManager;
		return -1;
	}

	// 学習, テスト実行
	{
		time_t startTime = time(NULL);

		// 学習
		if(::LearnWithCalculateSampleError(pNeuralNetwork, *pLayerDataSOM, USE_BATCH_SIZE, MAX_EPOCH) != ErrorCode::ERROR_CODE_NONE)
		{
			delete pNeuralNetwork;
			delete pLayerDataManager;
			delete pLayerDLLManager;

			return -1;
		}

		time_t endTime = time(NULL);

		printf("経過時間(s) : %ld\n", (endTime - startTime));
	}


	// バッファ開放
	delete pNeuralNetwork;
	delete pLayerDataManager;
	delete pLayerDLLManager;

	printf("Press any key to continue");
	getc(stdin);


	return 0;
}



Layer::Connect::ILayerConnectData* CreateNeuralNetwork(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct& inputDataStruct, const IODataStruct& outputDataStruct, Gravisbell::Layer::ILayerDataSOM** o_ppLayerDataSOM)
{
	// ニューラルネットワーク作成クラスを作成
	Gravisbell::Utility::NeuralNetworkLayer::INeuralNetworkMaker* pNetworkMaker = Gravisbell::Utility::NeuralNetworkLayer::CreateNeuralNetworkManaker(layerDLLManager, layerDataManager, &inputDataStruct, 1);

	// ニューラルネットワークを作成
	Layer::Connect::ILayerConnectData* pNeuralNetwork = pNetworkMaker->GetNeuralNetworkLayer();
	if(pNeuralNetwork == NULL)
		return NULL;


	// レイヤーを追加する
	if(pNeuralNetwork)
	{
		// 入力信号を直前レイヤーに設定
		Gravisbell::GUID lastLayerGUID = pNeuralNetwork->GetInputGUID();

		lastLayerGUID = pNetworkMaker->AddSOMLayer(lastLayerGUID, outputDataStruct.GetDataCount(), RESOLUTION_COUNT);

		*o_ppLayerDataSOM = dynamic_cast<Gravisbell::Layer::ILayerDataSOM*>(pNeuralNetwork->GetLayerDataByGUID(lastLayerGUID));

		// 出力レイヤー設定
		pNeuralNetwork->SetOutputLayerGUID(lastLayerGUID);
	}

	// 出力データ構造が正しいことを確認
	if(pNeuralNetwork->GetOutputDataStruct(&inputDataStruct, 1) != outputDataStruct)
	{
		layerDataManager.EraseLayerByGUID(pNeuralNetwork->GetGUID());
		*o_ppLayerDataSOM = NULL;
		return NULL;
	}


	// オプティマイザーの設定
	pNeuralNetwork->ChangeOptimizer(L"Adam");

	delete pNetworkMaker;

	return pNeuralNetwork;
}


/** ニューラルネットワークの学習とサンプル実行を同時実行 */
Gravisbell::ErrorCode LearnWithCalculateSampleError(
	Layer::NeuralNetwork::INeuralNetwork* pNeuralNetwork,
	Gravisbell::Layer::ILayerDataSOM& layerDataSOM,
	const U32 BATCH_SIZE,
	const U32 LEARN_TIMES)
{
	Gravisbell::ErrorCode err;

	// 実行時設定
	pNeuralNetwork->SetRuntimeParameter(L"UseDropOut", true);

	pNeuralNetwork->SetRuntimeParameter(L"GaussianNoise_Bias", 0.0f);
	pNeuralNetwork->SetRuntimeParameter(L"GaussianNoise_Power", 0.0f);

	pNeuralNetwork->SetRuntimeParameter(L"SOM_ramda", 2500.0f);
	pNeuralNetwork->SetRuntimeParameter(L"SOM_sigma", 0.5f);

	// 事前処理を実行
	err = pNeuralNetwork->PreProcessLearn(BATCH_SIZE);
	if(err != ErrorCode::ERROR_CODE_NONE)
		return err;

	std::vector<F32> lpInputBuffer(pNeuralNetwork->GetInputBufferCount() * BATCH_SIZE);
	std::vector<F32> lpMapBuffer(layerDataSOM.GetMapSize());

	// ウィンドウと画像を作成
	cv::Mat img = cv::Mat::zeros(IMAGE_WIDTH, IMAGE_WIDTH, CV_8UC3);
	cv::namedWindow("drawing", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);


	// 学習を実行
	for(U32 learnTime=0; learnTime<LEARN_TIMES; learnTime++)
	{
		// 学習
		{
			// 学習ループ先頭処理
			pNeuralNetwork->PreProcessLoop();

			// 入力情報を更新
			for(U32 i=0; i<lpInputBuffer.size(); i++)
			{
				lpInputBuffer[i] = (float)(rand()%256) / 255.0f;
			}

			// 演算
			pNeuralNetwork->Calculate(&lpInputBuffer[0]);

			// 誤差計算

			// 学習
			pNeuralNetwork->Training(NULL, NULL);

			// 画像更新
			if(learnTime%1 == 0)
			{
				printf("%4d回目\n", learnTime);

				layerDataSOM.GetMapBuffer(&lpMapBuffer[0]);

				for(U32 y=0; y<RESOLUTION_COUNT; y++)
				{
					for(U32 x=0; x<RESOLUTION_COUNT; x++)
					{
						U32 offset = (y*RESOLUTION_COUNT + x) * 3;

						int xPos = x * RECT_WIDTH;
						int yPos = y * RECT_WIDTH;

						U08 r = (std::min<float>(std::max<float>(lpMapBuffer[offset + 0], 0.0f), 1.0f) * 0xFF);
						U08 g = (std::min<float>(std::max<float>(lpMapBuffer[offset + 1], 0.0f), 1.0f) * 0xFF);
						U08 b = (std::min<float>(std::max<float>(lpMapBuffer[offset + 2], 0.0f), 1.0f) * 0xFF);

						cv::rectangle(img, cv::Point(xPos,yPos), cv::Point(xPos+RECT_WIDTH, yPos+RECT_WIDTH), cv::Scalar(r,g,b), -1, CV_AA);
					}
				}

				cv::imshow("drawing", img);
				cv::waitKey(500);
			}
		}

	}

	return ErrorCode::ERROR_CODE_NONE;
}