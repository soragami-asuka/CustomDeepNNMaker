// Sample04_MNIST.cpp : �R���\�[�� �A�v���P�[�V�����̃G���g�� �|�C���g���`���܂��B
//

#include "stdafx.h"
#include<crtdbg.h>

#include<vector>
#include<boost/filesystem/path.hpp>
#include<boost/uuid/uuid_generators.hpp>

#include"Library/Common/BatchDataNoListGenerator/BatchDataNoListGenerator.h"
#include"Library/DataFormat/Binary/DataFormat.h"
#include"Library/NeuralNetwork/LayerDLLManager/LayerDLLManager.h"
#include"Layer/IOData/IODataLayer/IODataLayer.h"
#include"Layer/NeuralNetwork/INNLayerConnectData.h"
#include"Layer/NeuralNetwork/INeuralNetwork.h"
#include"Utility/NeuralNetworkLayer/NeuralNetworkLayer.h"

using namespace Gravisbell;

#define USE_GPU	0
#define USE_HOST_MEMORY 1


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
Layer::NeuralNetwork::INNLayerConnectData* CreateNeuralNetwork(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, std::list<Layer::ILayerData*>& lppLayerData, const IODataStruct& inputDataStruct, const IODataStruct& outputDataStruct);

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
		delete pDataLayerTeach_Output;
		delete pDataLayerTest_Input;
		delete pDataLayerTest_Output;
		return -1;
	}

	// �j���[�����l�b�g���[�N�쐬
	std::list<Layer::ILayerData*> lppLayerData;
	Gravisbell::Layer::NeuralNetwork::INNLayerData* pNeuralNetworkData = CreateNeuralNetwork(*pLayerDLLManager, lppLayerData, pDataLayerTeach_Input->GetInputDataStruct(), pDataLayerTeach_Output->GetDataStruct());
	if(pNeuralNetworkData == NULL)
	{
		delete pDataLayerTeach_Input;
		delete pDataLayerTeach_Output;
		delete pDataLayerTest_Input;
		delete pDataLayerTest_Output;
		delete pLayerDLLManager;
		return -1;
	}

	// �w�K�p�j���[�����l�b�g���[�N�쐬
	Layer::NeuralNetwork::INeuralNetwork* pNeuralNetworkLearn = NULL;
	{
		Layer::NeuralNetwork::INNLayer* pLayer = pNeuralNetworkData->CreateLayer(boost::uuids::random_generator()().data);
		pNeuralNetworkLearn = dynamic_cast<Layer::NeuralNetwork::INeuralNetwork*>(pLayer);
		if(pNeuralNetworkLearn == NULL)
			delete pLayer;
	}
	if(pNeuralNetworkLearn == NULL)
	{
		for(auto pLayerData : lppLayerData)
			delete pLayerData;
		delete pDataLayerTeach_Input;
		delete pDataLayerTeach_Output;
		delete pDataLayerTest_Input;
		delete pDataLayerTest_Output;
		delete pLayerDLLManager;
		return -1;
	}

	// �e�X�g�p�j���[�����l�b�g���[�N�쐬
	Layer::NeuralNetwork::INeuralNetwork* pNeuralNetworkTest = NULL;
	{
		Layer::NeuralNetwork::INNLayer* pLayer = pNeuralNetworkData->CreateLayer(boost::uuids::random_generator()().data);
		pNeuralNetworkTest = dynamic_cast<Layer::NeuralNetwork::INeuralNetwork*>(pLayer);
		if(pNeuralNetworkTest == NULL)
			delete pLayer;
	}
	if(pNeuralNetworkTest == NULL)
	{
		delete pNeuralNetworkLearn;
		for(auto pLayerData : lppLayerData)
			delete pLayerData;
		delete pDataLayerTeach_Input;
		delete pDataLayerTeach_Output;
		delete pDataLayerTest_Input;
		delete pDataLayerTest_Output;
		delete pLayerDLLManager;
		return -1;
	}


	// �w�K, �e�X�g���s
	{
		// �w�K
		if(::LearnWithCalculateSampleError(pNeuralNetworkLearn, pNeuralNetworkTest, pDataLayerTeach_Input, pDataLayerTeach_Output, pDataLayerTest_Input, pDataLayerTest_Output, 8, 100) != ErrorCode::ERROR_CODE_NONE)
		{
			delete pNeuralNetworkLearn;
			delete pNeuralNetworkTest;
			for(auto pLayerData : lppLayerData)
				delete pLayerData;
			delete pDataLayerTeach_Input;
			delete pDataLayerTeach_Output;
			delete pDataLayerTest_Input;
			delete pDataLayerTest_Output;
			delete pLayerDLLManager;

			return -1;
		}

	}


	// �o�b�t�@�J��
	delete pNeuralNetworkLearn;
	delete pNeuralNetworkTest;
	for(auto pLayerData : lppLayerData)
		delete pLayerData;
	delete pDataLayerTeach_Input;
	delete pDataLayerTeach_Output;
	delete pDataLayerTest_Input;
	delete pDataLayerTest_Output;
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


/** �j���[�����l�b�g���[�N�N���X���쐬���� */
Layer::NeuralNetwork::INNLayerConnectData* CreateNeuralNetwork(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, std::list<Layer::ILayerData*>& lppLayerData, const IODataStruct& inputDataStruct, const IODataStruct& outputDataStruct)
{
	// �j���[�����l�b�g���[�N���쐬
	Layer::NeuralNetwork::INNLayerConnectData* pNeuralNetwork = NULL;
	{
		// DLL�擾
		const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(Gravisbell::GUID(0x1c38e21f, 0x6f01, 0x41b2, 0xb4, 0x0e, 0x7f, 0x67, 0x26, 0x7a, 0x36, 0x92));
		if(pLayerDLL == NULL)
			return NULL;

		// �ݒ�̍쐬
		SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
		if(pConfig == NULL)
			return NULL;

		// ���C���[�̍쐬
		Layer::NeuralNetwork::INNLayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, inputDataStruct);
		if(pLayer == NULL)
			return NULL;
		lppLayerData.push_back(pLayer);

		// �ݒ�����폜
		delete pConfig;

		// �L���X�g
		pNeuralNetwork = dynamic_cast<Layer::NeuralNetwork::INNLayerConnectData*>(pLayer);
		if(pNeuralNetwork == NULL)
		{
			delete pLayer;
			return NULL;
		}
	}

	// ���C���[��ǉ�����
	{
		// ���͐M���𒼑O���C���[�ɐݒ�
		Gravisbell::IODataStruct inputDataStruct = pNeuralNetwork->GetInputDataStruct();
		Gravisbell::GUID lastLayerGUID = pNeuralNetwork->GetInputGUID();

		// 1�w��
		Gravisbell::Utility::NeuralNetworkLayer::AddLayerToNetworkLast(
			*pNeuralNetwork,
			lppLayerData,
			inputDataStruct, lastLayerGUID,
			Gravisbell::Utility::NeuralNetworkLayer::CreateConvolutionLayer(layerDLLManager, inputDataStruct, Vector3D<S32>(4,4,1), 4, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0)));
		Gravisbell::Utility::NeuralNetworkLayer::AddLayerToNetworkLast(
			*pNeuralNetwork,
			lppLayerData,
			inputDataStruct, lastLayerGUID,
			Gravisbell::Utility::NeuralNetworkLayer::CreatePoolingLayer(layerDLLManager, inputDataStruct, Vector3D<S32>(2,2,1)));
		Gravisbell::Utility::NeuralNetworkLayer::AddLayerToNetworkLast(
			*pNeuralNetwork,
			lppLayerData,
			inputDataStruct, lastLayerGUID,
			Gravisbell::Utility::NeuralNetworkLayer::CreateBatchNormalizationLayer(layerDLLManager, inputDataStruct));
		Gravisbell::Utility::NeuralNetworkLayer::AddLayerToNetworkLast(
			*pNeuralNetwork,
			lppLayerData,
			inputDataStruct, lastLayerGUID,
			Gravisbell::Utility::NeuralNetworkLayer::CreateActivationLayer(layerDLLManager, inputDataStruct, L"ReLU"));

		// 2�w��
		Gravisbell::Utility::NeuralNetworkLayer::AddLayerToNetworkLast(
			*pNeuralNetwork,
			lppLayerData,
			inputDataStruct, lastLayerGUID,
			Gravisbell::Utility::NeuralNetworkLayer::CreateConvolutionLayer(layerDLLManager, inputDataStruct, Vector3D<S32>(4,4,1), 8, Vector3D<S32>(1,1,1), Vector3D<S32>(2,2,0)));
		Gravisbell::Utility::NeuralNetworkLayer::AddLayerToNetworkLast(
			*pNeuralNetwork,
			lppLayerData,
			inputDataStruct, lastLayerGUID,
			Gravisbell::Utility::NeuralNetworkLayer::CreatePoolingLayer(layerDLLManager, inputDataStruct, Vector3D<S32>(2,2,1)));
		Gravisbell::Utility::NeuralNetworkLayer::AddLayerToNetworkLast(
			*pNeuralNetwork,
			lppLayerData,
			inputDataStruct, lastLayerGUID,
			Gravisbell::Utility::NeuralNetworkLayer::CreateBatchNormalizationLayer(layerDLLManager, inputDataStruct));
		Gravisbell::Utility::NeuralNetworkLayer::AddLayerToNetworkLast(
			*pNeuralNetwork,
			lppLayerData,
			inputDataStruct, lastLayerGUID,
			Gravisbell::Utility::NeuralNetworkLayer::CreateActivationLayer(layerDLLManager, inputDataStruct, L"ReLU"));

		// 3�w��
		Gravisbell::Utility::NeuralNetworkLayer::AddLayerToNetworkLast(
			*pNeuralNetwork,
			lppLayerData,
			inputDataStruct, lastLayerGUID,
			Gravisbell::Utility::NeuralNetworkLayer::CreateFullyConnectLayer(layerDLLManager, inputDataStruct, outputDataStruct.GetDataCount()));
		Gravisbell::Utility::NeuralNetworkLayer::AddLayerToNetworkLast(
			*pNeuralNetwork,
			lppLayerData,
			inputDataStruct, lastLayerGUID,
			Gravisbell::Utility::NeuralNetworkLayer::CreateActivationLayer(layerDLLManager, inputDataStruct, L"softmax_ALL_crossEntropy"));

		// �o�̓��C���[�ݒ�
		pNeuralNetwork->SetOutputLayerGUID(lastLayerGUID);
	}

	return pNeuralNetwork;
}



/** �j���[�����l�b�g���[�N�̊w�K�ƃT���v�����s�𓯎����s */
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

	// �w�K�W����ݒ�
	pNeuralNetworkLearn->SetLearnSettingData(L"LearnCoeff", 0.001f);
//	pNeuralNetworkLearn->SetLearnSettingData(L"LearnCoeff", 0.005f);

	// ���O���������s
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


	// �o�b�`No�����N���X���쐬
	Gravisbell::Common::IBatchDataNoListGenerator* pBatchDataNoListGenerator = Gravisbell::Common::CreateBatchDataNoListGenerator();
	err = pBatchDataNoListGenerator->PreProcess(pTeachInputLayer->GetDataCount(), BATCH_SIZE);
	if(err != ErrorCode::ERROR_CODE_NONE)
	{
		delete pBatchDataNoListGenerator;
		return err;
	}

	// �_�~�[�̊w�K�ݒ���쐬
	Gravisbell::SettingData::Standard::IData* pLearnSetting = Gravisbell::Layer::IOData::CreateLearningSetting();
	if(pLearnSetting == NULL)
	{
		delete pBatchDataNoListGenerator;
		return err;
	}

	std::vector<F32> lpOutputBuffer(pTeachOutputLayer->GetBufferCount() * BATCH_SIZE);
	std::vector<F32> lpTeachBuffer(pTeachOutputLayer->GetBufferCount() * BATCH_SIZE);

	// �w�K�����s
	for(U32 learnTime=0; learnTime<LEARN_TIMES; learnTime++)
	{
//		printf("%5d�� ", learnTime);
		printf("%5d,", learnTime);

		U32 correctCount_learn  = 0;	// ����
		U32 correctCount_sample = 0;	// ����

		// �w�K
		{
			// �w�K���[�v�擪����
			pBatchDataNoListGenerator->PreProcessLearnLoop();
			pTeachInputLayer->PreProcessLearnLoop(*pLearnSetting);
			pTeachOutputLayer->PreProcessLearnLoop(*pLearnSetting);
			pNeuralNetworkLearn->PreProcessLearnLoop(*pLearnSetting);

			// �w�K����
			// �o�b�`�P�ʂŏ���
			for(U32 batchNum=0; batchNum<pBatchDataNoListGenerator->GetBatchDataNoListCount(); batchNum++)
			{
				printf(" L=%5.1f%%", (F32)batchNum * 100 / pBatchDataNoListGenerator->GetBatchDataNoListCount());
				printf("\b\b\b\b\b\b\b\b\b");

				// �f�[�^�؂�ւ�
				pTeachInputLayer->SetBatchDataNoList(pBatchDataNoListGenerator->GetBatchDataNoListByNum(batchNum));
				pTeachOutputLayer->SetBatchDataNoList(pBatchDataNoListGenerator->GetBatchDataNoListByNum(batchNum));

				// ���Z
				pNeuralNetworkLearn->Calculate(pTeachInputLayer->GetOutputBuffer());

				// �덷�v�Z
				// ���t�M���Ƃ̌덷�v�Z
				pTeachOutputLayer->CalculateLearnError(pNeuralNetworkLearn->GetOutputBuffer());
				pNeuralNetworkLearn->CalculateLearnError(pTeachOutputLayer->GetDInputBuffer());

				// �덷�𔽉f
				pNeuralNetworkLearn->ReflectionLearnError();

				// ���𗦂��Z�o����
				pTeachOutputLayer->GetOutputBuffer(&lpTeachBuffer[0]);
				pNeuralNetworkLearn->GetOutputBuffer(&lpOutputBuffer[0]);
				for(U32 batchDataNum=0; batchDataNum<pTeachOutputLayer->GetBatchSize(); batchDataNum++)
				{
					// �����̔ԍ����擾
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
					// �o�͂��ꂽ�ԍ����擾
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


		// �T���v�����s
		{		
			// �T���v�����s�擪����
			pSampleInputLayer->PreProcessCalculateLoop();
			pSampleOutputLayer->PreProcessCalculateLoop();
			pNeuralNetworkSample->PreProcessCalculateLoop();

			// �o�b�`�P�ʂŏ���
			for(U32 dataNum=0; dataNum<pSampleInputLayer->GetDataCount(); dataNum++)
			{
				printf(" T=%5.1f%%", (F32)dataNum * 100 / pSampleInputLayer->GetDataCount());
				printf("\b\b\b\b\b\b\b\b\b");

				// �f�[�^�؂�ւ�
				pSampleInputLayer->SetBatchDataNoList(&dataNum);
				pSampleOutputLayer->SetBatchDataNoList(&dataNum);

				// ���Z
				pNeuralNetworkSample->Calculate(pSampleInputLayer->GetOutputBuffer());

				// �덷�v�Z
				pSampleOutputLayer->CalculateLearnError(pNeuralNetworkSample->GetOutputBuffer());


				// �����̔ԍ����擾
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
					// �o�͂��ꂽ�ԍ����擾
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

		// �덷�\��
		{
			F32 errorMin, errorMax, errorAve, errorAve2, errorCrossEntoropy;
			pTeachOutputLayer->GetCalculateErrorValue(errorMin, errorMax, errorAve, errorAve2, errorCrossEntoropy);
//			printf("�w�K�Fmax=%.3f, ave=%.3f, ave2=%.3f, entropy=%.3f", errorMax, errorAve, errorAve2, errorCrossEntoropy);
			printf("%.3f,%.3f,%.3f,%.3f,",  errorMax, errorAve2, errorCrossEntoropy, (F32)correctCount_learn / (pBatchDataNoListGenerator->GetBatchDataNoListCount() * BATCH_SIZE)); 
		}
//		printf(" : ");
		{
			F32 errorMin, errorMax, errorAve, errorAve2, errorCrossEntoropy;
			pSampleOutputLayer->GetCalculateErrorValue(errorMin, errorMax, errorAve, errorAve2, errorCrossEntoropy);
//			printf("���s�Fmax=%.3f, ave=%.3f, ave2=%.3f, entropy=%.3f", errorMax, errorAve, errorAve2, errorCrossEntoropy);
			printf("%.3f,%.3f,%.3f,%.3f", errorMax, errorAve2, errorCrossEntoropy, (F32)correctCount_sample / pSampleInputLayer->GetDataCount()); 
		}
		printf("\n");
	}

	// �������J��
	delete pLearnSetting;
	delete pBatchDataNoListGenerator;

	return ErrorCode::ERROR_CODE_NONE;
}