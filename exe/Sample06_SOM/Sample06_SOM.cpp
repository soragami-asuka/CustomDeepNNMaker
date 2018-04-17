//=============================================
// �I�[�g�G���R�[�_���������邽�߂̃T���v��
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


/** �j���[�����l�b�g���[�N�N���X���쐬���� */
Layer::Connect::ILayerConnectData* CreateNeuralNetwork(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct& inputDataStruct, const IODataStruct& outputDataStruct, Gravisbell::Layer::ILayerDataSOM** o_ppLayerDataSOM);


/** �j���[�����l�b�g���[�N�̊w�K�ƃT���v�����s�𓯎����s */
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


	// ���C���[DLL�Ǘ��N���X���쐬
#if USE_GPU
	Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager = Gravisbell::Utility::NeuralNetworkLayer::CreateLayerDLLManagerGPU(L"./");
#else
	Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager = Gravisbell::Utility::NeuralNetworkLayer::CreateLayerDLLManagerCPU(L"./");
#endif
	if(pLayerDLLManager == NULL)
	{
		return -1;
	}

	// ���C���[�f�[�^�Ǘ��N���X���쐬
	Gravisbell::Layer::NeuralNetwork::ILayerDataManager* pLayerDataManager = Gravisbell::Layer::NeuralNetwork::CreateLayerDataManager();
	if(pLayerDataManager == NULL)
	{
		delete pLayerDLLManager;
		return -1;
	}

	// �������Œ�
#ifdef _DEBUG
//	Gravisbell::Layer::NeuralNetwork::GetInitializerManager().InitializeRandomParameter(0);
#endif

	// �G���R�[�_0���쐬
	Gravisbell::Layer::ILayerDataSOM* pLayerDataSOM = NULL;
	Gravisbell::Layer::Connect::ILayerConnectData* pNNData = CreateNeuralNetwork(*pLayerDLLManager, *pLayerDataManager, 3, 2, &pLayerDataSOM);
	if(pNNData == NULL)
	{
		delete pLayerDataManager;
		delete pLayerDLLManager;
		return -1;
	}


	// �w�K�p�j���[�����l�b�g���[�N�쐬
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

	// �w�K, �e�X�g���s
	{
		time_t startTime = time(NULL);

		// �w�K
		if(::LearnWithCalculateSampleError(pNeuralNetwork, *pLayerDataSOM, USE_BATCH_SIZE, MAX_EPOCH) != ErrorCode::ERROR_CODE_NONE)
		{
			delete pNeuralNetwork;
			delete pLayerDataManager;
			delete pLayerDLLManager;

			return -1;
		}

		time_t endTime = time(NULL);

		printf("�o�ߎ���(s) : %ld\n", (endTime - startTime));
	}


	// �o�b�t�@�J��
	delete pNeuralNetwork;
	delete pLayerDataManager;
	delete pLayerDLLManager;

	printf("Press any key to continue");
	getc(stdin);


	return 0;
}



Layer::Connect::ILayerConnectData* CreateNeuralNetwork(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct& inputDataStruct, const IODataStruct& outputDataStruct, Gravisbell::Layer::ILayerDataSOM** o_ppLayerDataSOM)
{
	// �j���[�����l�b�g���[�N�쐬�N���X���쐬
	Gravisbell::Utility::NeuralNetworkLayer::INeuralNetworkMaker* pNetworkMaker = Gravisbell::Utility::NeuralNetworkLayer::CreateNeuralNetworkManaker(layerDLLManager, layerDataManager, &inputDataStruct, 1);

	// �j���[�����l�b�g���[�N���쐬
	Layer::Connect::ILayerConnectData* pNeuralNetwork = pNetworkMaker->GetNeuralNetworkLayer();
	if(pNeuralNetwork == NULL)
		return NULL;


	// ���C���[��ǉ�����
	if(pNeuralNetwork)
	{
		// ���͐M���𒼑O���C���[�ɐݒ�
		Gravisbell::GUID lastLayerGUID = pNeuralNetwork->GetInputGUID();

		lastLayerGUID = pNetworkMaker->AddSOMLayer(lastLayerGUID, outputDataStruct.GetDataCount(), RESOLUTION_COUNT);

		*o_ppLayerDataSOM = dynamic_cast<Gravisbell::Layer::ILayerDataSOM*>(pNeuralNetwork->GetLayerDataByGUID(lastLayerGUID));

		// �o�̓��C���[�ݒ�
		pNeuralNetwork->SetOutputLayerGUID(lastLayerGUID);
	}

	// �o�̓f�[�^�\�������������Ƃ��m�F
	if(pNeuralNetwork->GetOutputDataStruct(&inputDataStruct, 1) != outputDataStruct)
	{
		layerDataManager.EraseLayerByGUID(pNeuralNetwork->GetGUID());
		*o_ppLayerDataSOM = NULL;
		return NULL;
	}


	// �I�v�e�B�}�C�U�[�̐ݒ�
	pNeuralNetwork->ChangeOptimizer(L"Adam");

	delete pNetworkMaker;

	return pNeuralNetwork;
}


/** �j���[�����l�b�g���[�N�̊w�K�ƃT���v�����s�𓯎����s */
Gravisbell::ErrorCode LearnWithCalculateSampleError(
	Layer::NeuralNetwork::INeuralNetwork* pNeuralNetwork,
	Gravisbell::Layer::ILayerDataSOM& layerDataSOM,
	const U32 BATCH_SIZE,
	const U32 LEARN_TIMES)
{
	Gravisbell::ErrorCode err;

	// ���s���ݒ�
	pNeuralNetwork->SetRuntimeParameter(L"UseDropOut", true);

	pNeuralNetwork->SetRuntimeParameter(L"GaussianNoise_Bias", 0.0f);
	pNeuralNetwork->SetRuntimeParameter(L"GaussianNoise_Power", 0.0f);

	pNeuralNetwork->SetRuntimeParameter(L"SOM_ramda", 2500.0f);
	pNeuralNetwork->SetRuntimeParameter(L"SOM_sigma", 0.5f);

	// ���O���������s
	err = pNeuralNetwork->PreProcessLearn(BATCH_SIZE);
	if(err != ErrorCode::ERROR_CODE_NONE)
		return err;

	std::vector<F32> lpInputBuffer(pNeuralNetwork->GetInputBufferCount() * BATCH_SIZE);
	std::vector<F32> lpMapBuffer(layerDataSOM.GetMapSize());

	// �E�B���h�E�Ɖ摜���쐬
	cv::Mat img = cv::Mat::zeros(IMAGE_WIDTH, IMAGE_WIDTH, CV_8UC3);
	cv::namedWindow("drawing", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);


	// �w�K�����s
	for(U32 learnTime=0; learnTime<LEARN_TIMES; learnTime++)
	{
		// �w�K
		{
			// �w�K���[�v�擪����
			pNeuralNetwork->PreProcessLoop();

			// ���͏����X�V
			for(U32 i=0; i<lpInputBuffer.size(); i++)
			{
				lpInputBuffer[i] = (float)(rand()%256) / 255.0f;
			}

			// ���Z
			pNeuralNetwork->Calculate(&lpInputBuffer[0]);

			// �덷�v�Z

			// �w�K
			pNeuralNetwork->Training(NULL, NULL);

			// �摜�X�V
			if(learnTime%1 == 0)
			{
				printf("%4d���\n", learnTime);

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