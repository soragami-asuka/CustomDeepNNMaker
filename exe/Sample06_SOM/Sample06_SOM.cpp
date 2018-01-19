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
#include"Library/Layer/IOData/IODataLayer.h"
#include"Layer/Connect/ILayerConnectData.h"
#include"Layer/NeuralNetwork/INeuralNetwork.h"
#include"Utility/NeuralNetworkLayer.h"
#include"Utility/NeuralNetworkMaker.h"

#include"Library/NeuralNetwork/Initializer.h"

using namespace Gravisbell;

#define USE_GPU	0
#define USE_HOST_MEMORY 1

#define USE_BATCH_SIZE	128
#define MAX_EPOCH		20

#define RESOLUTION_COUNT	(16)


/** �f�[�^�t�@�C������ǂݍ���
	@param	o_ppDataLayerTeach	���t�f�[�^���i�[�����f�[�^�N���X�̊i�[��|�C���^�A�h���X
	@param	o_ppDataLayerTest	�e�X�g�f�[�^���i�[�����f�[�^�N���X�̊i�[��|�C���^�A�h���X
	@param	i_testRate			�e�X�g�f�[�^��S�̂̉�%�ɂ��邩0�`1�̊ԂŐݒ�
	@param	i_formatFilePath	�t�H�[�}�b�g�ݒ�̓�����XML�t�@�C���p�X
	@param	i_dataFilePath		�f�[�^�̓������o�C�i���t�@�C���p�X
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


/** �j���[�����l�b�g���[�N�N���X���쐬���� */
Layer::Connect::ILayerConnectData* CreateNeuralNetwork(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct& inputDataStruct, const IODataStruct& outputDataStruct);


/** �j���[�����l�b�g���[�N�̊w�K�ƃT���v�����s�𓯎����s */
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

	// �摜��ǂݍ���
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


	// ���C���[DLL�Ǘ��N���X���쐬
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

	// ���C���[�f�[�^�Ǘ��N���X���쐬
	Gravisbell::Layer::NeuralNetwork::ILayerDataManager* pLayerDataManager = Gravisbell::Layer::NeuralNetwork::CreateLayerDataManager();
	if(pLayerDataManager == NULL)
	{
		delete pDataLayerTeach_Input;
		delete pDataLayerTest_Input;
		delete pDataLayerTest_Output;
		delete pLayerDLLManager;
		return -1;
	}

	// �������Œ�
#ifdef _DEBUG
//	Gravisbell::Layer::NeuralNetwork::GetInitializerManager().InitializeRandomParameter(0);
#endif

	// �G���R�[�_0���쐬
	Gravisbell::Layer::Connect::ILayerConnectData* pNNData = CreateNeuralNetwork(*pLayerDLLManager, *pLayerDataManager, pDataLayerTeach_Input->GetInputDataStruct(), 2);
	if(pNNData == NULL)
	{
		delete pDataLayerTeach_Input;
		delete pDataLayerTest_Input;
		delete pDataLayerTest_Output;
		delete pLayerDataManager;
		delete pLayerDLLManager;
		return -1;
	}


	// �w�K�p�j���[�����l�b�g���[�N�쐬
	Layer::NeuralNetwork::INeuralNetwork* pAutoencoder_learn = NULL;
	{
#if USE_HOST_MEMORY
		Layer::ILayerBase* pLayer = pNNData->CreateLayer(boost::uuids::random_generator()().data, &pDataLayerTeach_Input->GetOutputDataStruct(), 1);
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

	// �e�X�g�p�j���[�����l�b�g���[�N�쐬
	Layer::NeuralNetwork::INeuralNetwork* pAutoencoder_test = NULL;
	{
#if USE_HOST_MEMORY
		Layer::ILayerBase* pLayer = pNNData->CreateLayer(boost::uuids::random_generator()().data, &pDataLayerTeach_Input->GetOutputDataStruct(), 1);
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

	// �w�K, �e�X�g���s
	{
		time_t startTime = time(NULL);

		// �w�K
		if(::LearnWithCalculateSampleError(pAutoencoder_learn, pAutoencoder_test, pDataLayerTeach_Input, pDataLayerTeach_Output, pDataLayerTest_Input, pDataLayerTest_Output, USE_BATCH_SIZE, MAX_EPOCH) != ErrorCode::ERROR_CODE_NONE)
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

		printf("�o�ߎ���(s) : %ld\n", (endTime - startTime));
	}


	// �o�b�t�@�J��
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


/** �f�[�^�t�@�C������ǂݍ���
	@param	o_ppDataLayerTeach	���t�f�[�^���i�[�����f�[�^�N���X�̊i�[��|�C���^�A�h���X
	@param	o_ppDataLayerTest	�e�X�g�f�[�^���i�[�����f�[�^�N���X�̊i�[��|�C���^�A�h���X
	@param	i_testRate			�e�X�g�f�[�^��S�̂̉�%�ɂ��邩0�`1�̊ԂŐݒ�
	@param	i_formatFilePath	�t�H�[�}�b�g�ݒ�̓�����XML�t�@�C���p�X
	@param	i_dataFilePath		�f�[�^�̓������o�C�i���t�@�C���p�X
	*/
Gravisbell::ErrorCode LoadSampleData_image(
	Layer::IOData::IIODataLayer** o_ppDataLayerTeach,  Layer::IOData::IIODataLayer** o_ppDataLayerTest,
	F32 i_testRate,
	boost::filesystem::wpath i_formatFilePath,
	boost::filesystem::wpath i_dataFilePath)
{
	// �t�H�[�}�b�g��ǂݍ���
	Gravisbell::DataFormat::Binary::IDataFormat* pDataFormat = Gravisbell::DataFormat::Binary::CreateDataFormatFromXML(i_formatFilePath.c_str());
	if(pDataFormat == NULL)
		return Gravisbell::ErrorCode::ERROR_CODE_COMMON_FILE_NOT_FOUND;

	// �o�b�t�@��ǂݍ���
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

	// �t�H�[�}�b�g���g���ăw�b�_��ǂݍ���
	U32 bufPos = 0;

	// �w�b�_��ǂݍ���
	bufPos = pDataFormat->LoadBinary(&lpBuf[0], (U32)lpBuf.size());

	// �f�[�^�\�����쐬����
	Gravisbell::IODataStruct dataStruct(1, pDataFormat->GetVariableValue(L"columns"), pDataFormat->GetVariableValue(L"rows"), 1);


	*o_ppDataLayerTeach = Gravisbell::Layer::IOData::CreateIODataLayerCPU(dataStruct);
	*o_ppDataLayerTest  = Gravisbell::Layer::IOData::CreateIODataLayerCPU(dataStruct);


	std::vector<F32> lpTmpBuf(dataStruct.GetDataCount());

	// �f�[�^�̌�����
	U32 dataCount = (U32)pDataFormat->GetVariableValue(L"images");
	U32 teachDataCount = (U32)(dataCount*(1.0f - i_testRate));
	for(U32 imageNum=0; imageNum<dataCount; imageNum++)
	{
		if(bufPos + dataStruct.GetDataCount() > lpBuf.size())
			break;

		// U08 -> F32 �ϊ�
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

	// �f�[�^�t�H�[�}�b�g�폜
	delete pDataFormat;

	return Gravisbell::ErrorCode::ERROR_CODE_NONE;
}
Gravisbell::ErrorCode LoadSampleData_label(
	Layer::IOData::IIODataLayer** o_ppDataLayerTeach,  Layer::IOData::IIODataLayer** o_ppDataLayerTest,
	F32 i_testRate,
	boost::filesystem::wpath i_formatFilePath,
	boost::filesystem::wpath i_dataFilePath)
{
	// �t�H�[�}�b�g��ǂݍ���
	Gravisbell::DataFormat::Binary::IDataFormat* pDataFormat = Gravisbell::DataFormat::Binary::CreateDataFormatFromXML(i_formatFilePath.c_str());
	if(pDataFormat == NULL)
		return Gravisbell::ErrorCode::ERROR_CODE_COMMON_FILE_NOT_FOUND;

	// �o�b�t�@��ǂݍ���
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

	// �t�H�[�}�b�g���g���ăw�b�_��ǂݍ���
	U32 bufPos = 0;

	// �w�b�_��ǂݍ���
	bufPos = pDataFormat->LoadBinary(&lpBuf[0], (U32)lpBuf.size());

	// �f�[�^�\�����쐬����
	Gravisbell::IODataStruct dataStruct(10, 1, 1, 1);


	*o_ppDataLayerTeach = Gravisbell::Layer::IOData::CreateIODataLayerCPU(dataStruct);
	*o_ppDataLayerTest  = Gravisbell::Layer::IOData::CreateIODataLayerCPU(dataStruct);


	std::vector<F32> lpTmpBuf(dataStruct.ch);

	// �f�[�^�̌�����
	U32 dataCount = (U32)pDataFormat->GetVariableValue(L"images");
	U32 teachDataCount = (U32)(dataCount*(1.0f - i_testRate));
	for(U32 imageNum=0; imageNum<dataCount; imageNum++)
	{
		// U08 -> F32 �ϊ�
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

	// �f�[�^�t�H�[�}�b�g�폜
	delete pDataFormat;

	return Gravisbell::ErrorCode::ERROR_CODE_NONE;
}



Layer::Connect::ILayerConnectData* CreateNeuralNetwork(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct& inputDataStruct, const IODataStruct& outputDataStruct)
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

		lastLayerGUID = pNetworkMaker->AddSOMLayer(lastLayerGUID, outputDataStruct.GetDataCount(), 16);

		// �o�̓��C���[�ݒ�
		pNeuralNetwork->SetOutputLayerGUID(lastLayerGUID);
	}

	// �o�̓f�[�^�\�������������Ƃ��m�F
	if(pNeuralNetwork->GetOutputDataStruct(&inputDataStruct, 1) != outputDataStruct)
	{
		layerDataManager.EraseLayerByGUID(pNeuralNetwork->GetGUID());
		return NULL;
	}


	// �I�v�e�B�}�C�U�[�̐ݒ�
	pNeuralNetwork->ChangeOptimizer(L"Adam");

	delete pNetworkMaker;

	return pNeuralNetwork;
}


/** �j���[�����l�b�g���[�N�̊w�K�ƃT���v�����s�𓯎����s */
Gravisbell::ErrorCode LearnWithCalculateSampleError(
	Layer::NeuralNetwork::INeuralNetwork* pNeuralNetworkLearn,
	Layer::NeuralNetwork::INeuralNetwork* pNeuralNetworkSample,
	Layer::IOData::IIODataLayer* pTeachInputLayer,
	Layer::IOData::IIODataLayer* pTeachTeachLayer,
	Layer::IOData::IIODataLayer* pSampleInputLayer,
	Layer::IOData::IIODataLayer* pSampleTeachLayer,
	const U32 BATCH_SIZE,
	const U32 LEARN_TIMES)
{
	Gravisbell::ErrorCode err;

	// ���s���ݒ�
	pNeuralNetworkLearn->SetRuntimeParameter(L"UseDropOut", true);
	pNeuralNetworkSample->SetRuntimeParameter(L"UseDropOut", false);
	
	pNeuralNetworkLearn->SetRuntimeParameter(L"GaussianNoise_Bias", 0.0f);
	pNeuralNetworkLearn->SetRuntimeParameter(L"GaussianNoise_Power", 0.0f);
	pNeuralNetworkSample->SetRuntimeParameter(L"GaussianNoise_Bias", 0.0f);
	pNeuralNetworkSample->SetRuntimeParameter(L"GaussianNoise_Power", 0.0f);

	pNeuralNetworkLearn->SetRuntimeParameter(L"SOM_ramda", 2500.0f);
	pNeuralNetworkLearn->SetRuntimeParameter(L"SOM_sigma", 1.0f);

	// ���O���������s
	err = pNeuralNetworkLearn->PreProcessLearn(BATCH_SIZE);
	if(err != ErrorCode::ERROR_CODE_NONE)
		return err;
	err = pTeachInputLayer->PreProcessLearn(BATCH_SIZE);
	if(err != ErrorCode::ERROR_CODE_NONE)
		return err;
	err = pTeachTeachLayer->PreProcessLearn(BATCH_SIZE);
	if(err != ErrorCode::ERROR_CODE_NONE)
		return err;


	err = pNeuralNetworkSample->PreProcessCalculate(1);
	if(err != ErrorCode::ERROR_CODE_NONE)
		return err;
	err = pSampleInputLayer->PreProcessCalculate(1);
	if(err != ErrorCode::ERROR_CODE_NONE)
		return err;


	// �o�b�`No�����N���X���쐬
	Gravisbell::Common::IBatchDataNoListGenerator* pBatchDataNoListGenerator = Gravisbell::Common::CreateBatchDataNoListGenerator();
	err = pBatchDataNoListGenerator->PreProcess(pTeachInputLayer->GetDataCount(), BATCH_SIZE);
	if(err != ErrorCode::ERROR_CODE_NONE)
	{
		delete pBatchDataNoListGenerator;
		return err;
	}


	// �w�K�����s
	for(U32 learnTime=0; learnTime<LEARN_TIMES; learnTime++)
	{
		printf("%5d,", learnTime);

		std::vector<std::vector<std::vector<U32>>> lpProtCount(10);
		for(U32 no=0; no<lpProtCount.size(); no++)
		{
			lpProtCount[no].resize(RESOLUTION_COUNT);
			for(U32 y=0; y<RESOLUTION_COUNT; y++)
			{
				lpProtCount[no][y].resize(RESOLUTION_COUNT, 0);
			}
		}

		// �w�K
		{
			// �w�K���[�v�擪����
//			pBatchDataNoListGenerator->PreProcessLearnLoop();
			pTeachInputLayer->PreProcessLoop();
			pTeachTeachLayer->PreProcessLoop();
			pNeuralNetworkLearn->PreProcessLoop();


			// �w�K����
			// �o�b�`�P�ʂŏ���
			for(U32 batchNum=0; batchNum<pBatchDataNoListGenerator->GetBatchDataNoListCount(); batchNum++)
			{
#if USE_GPU
				if(batchNum%10 == 0)
#endif
				{
					printf(" L=%5.1f%%", (F32)batchNum * 100 / pBatchDataNoListGenerator->GetBatchDataNoListCount());
					printf("\b\b\b\b\b\b\b\b\b");
				}

				// �f�[�^�؂�ւ�
				pTeachInputLayer->SetBatchDataNoList(pBatchDataNoListGenerator->GetBatchDataNoListByNum(batchNum));
				pTeachTeachLayer->SetBatchDataNoList(pBatchDataNoListGenerator->GetBatchDataNoListByNum(batchNum));

				// ���Z
				pNeuralNetworkLearn->Calculate(pTeachInputLayer->GetOutputBuffer());

				// �덷�v�Z

				// �w�K
				pNeuralNetworkLearn->Training(NULL, NULL);


				for(U32 dataNo=0; dataNo<BATCH_SIZE; dataNo++)
				{
					U32 no = 0;
					{
						F32 maxValue = pTeachTeachLayer->GetOutputBuffer()[10*dataNo + 0];
						for(int i=1; i<10; i++)
						{
							if(pTeachTeachLayer->GetOutputBuffer()[10*dataNo + i] > maxValue)
							{
								no = i;
								maxValue = pTeachTeachLayer->GetOutputBuffer()[10*dataNo + i];
							}
						}
					}

					float y = pNeuralNetworkLearn->GetOutputBuffer()[2*dataNo + 0];
					float x = pNeuralNetworkLearn->GetOutputBuffer()[2*dataNo + 1];

					U32 yPos = (U32)(y * (RESOLUTION_COUNT-1));
					U32 xPos = (U32)(x * (RESOLUTION_COUNT-1));

					lpProtCount[no][yPos][xPos]++;
				}
			}
		}


		// �T���v�����s
//		{		
//			// �T���v�����s�擪����
//			pSampleInputLayer->PreProcessLoop();
//			pNeuralNetworkSample->PreProcessLoop();
//
//			// �o�b�`�P�ʂŏ���
//			for(U32 dataNum=0; dataNum<pSampleInputLayer->GetDataCount(); dataNum++)
//			{
//#if USE_GPU
//				if(dataNum%10 == 0)
//#endif
//				{
//					printf(" T=%5.1f%%", (F32)dataNum * 100 / pSampleInputLayer->GetDataCount());
//					printf("\b\b\b\b\b\b\b\b\b");
//				}
//
//				// �f�[�^�؂�ւ�
//				pSampleInputLayer->SetBatchDataNoList(&dataNum);
//
//				// ���Z
//				pNeuralNetworkSample->Calculate(pSampleInputLayer->GetOutputBuffer());
//
//				// �덷�v�Z
//			}
//		}

		// �v���b�g�񐔂���������
		{
			char szFileName[256];
			sprintf(szFileName, "log[%03d].csv", learnTime);

			FILE* fp = fopen(szFileName, "w");
			for(U32 no=0; no<lpProtCount.size(); no++)
			{
				fprintf(fp, "No,%d\n", no);
				for(U32 y=0; y<RESOLUTION_COUNT; y++)
				{
					for(U32 x=0; x<RESOLUTION_COUNT; x++)
					{
						fprintf(fp, "%d,", lpProtCount[no][y][x]);
					}
					fprintf(fp, "\n");
				}
				fprintf(fp, "\n");
			}
			fclose(fp);
		}


		// �덷�\��
		{
			F32 errorMax, errorAve, errorAve2, errorCrossEntoropy;
			pTeachInputLayer->GetCalculateErrorValue(errorMax, errorAve, errorAve2, errorCrossEntoropy);
//			printf("�w�K�Fmax=%.3f, ave=%.3f, ave2=%.3f, entropy=%.3f", errorMax, errorAve, errorAve2, errorCrossEntoropy);
			printf("%.3f,%.3f,%.3f - ",  errorMax, errorAve2, errorCrossEntoropy); 
		}
//		printf(" : ");
		{
			F32 errorMax, errorAve, errorAve2, errorCrossEntoropy;
			pSampleInputLayer->GetCalculateErrorValue(errorMax, errorAve, errorAve2, errorCrossEntoropy);
//			printf("���s�Fmax=%.3f, ave=%.3f, ave2=%.3f, entropy=%.3f", errorMax, errorAve, errorAve2, errorCrossEntoropy);
			printf("%.3f,%.3f,%.3f", errorMax, errorAve2, errorCrossEntoropy); 
		}
		printf("\n");
	}

	// �������J��
	delete pBatchDataNoListGenerator;

	return ErrorCode::ERROR_CODE_NONE;
}