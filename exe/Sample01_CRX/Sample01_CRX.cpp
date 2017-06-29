//=============================================
// �N���W�b�g�J�[�h�F�؂̃f�[�^��p���������T���v��
// �Q�lURL�F
// �EDropout�F�f�B�[�v���[�j���O�̉Εt�����A�P���ȕ��@�ŉߊw�K��h��
//	https://deepage.net/deep_learning/2016/10/17/deeplearning_dropout.html
//
// �T���v���f�[�^URL:
// https://archive.ics.uci.edu/ml/datasets/Credit+Approval
//  �f�[�^�{��
//		Data Folder > crx.data
//  �f�[�^�t�H�[�}�b�g�ɂ���
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

/** �T���v���f�[�^�̓ǂݍ��� */
DataFormat::IDataFormatBase* LoadSampleData(const std::wstring& formatFilePath, const std::wstring& dataFilePath);

/** �j���[�����l�b�g���[�N�N���X���쐬���� */
Layer::Connect::ILayerConnectData* CreateNeuralNetwork(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	const IODataStruct& inputDataStruct, const IODataStruct& outputDataStruct);

/** �j���[�����l�b�g���[�N�̊w�K */
Gravisbell::ErrorCode LearnNeuralNetwork(
	Layer::NeuralNetwork::INeuralNetwork* pNeuralNetwork,
	Layer::IOData::IIODataLayer* pInputLayer,
	Layer::IOData::IIODataLayer* pTeachLayer,
	const U32 BATCH_SIZE,
	const U32 LEARN_TIMES);

/** �T���v���f�[�^�Ƃ̌덷�v�� */
Gravisbell::ErrorCode CalculateSampleError(
	Layer::NeuralNetwork::INeuralNetwork* pNeuralNetwork,
	Layer::IOData::IIODataLayer* pInputLayer,
	Layer::IOData::IIODataLayer* pTeachLayer);

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


	srand(12345);

	// �T���v���f�[�^�̓ǂݍ���
#ifndef _WIN64
	Gravisbell::DataFormat::IDataFormatBase* pSampleData = ::LoadSampleData(L"../SampleData/CRX/DataFormat.xml", L"../SampleData/CRX/crx.csv");
#else
	Gravisbell::DataFormat::IDataFormatBase* pSampleData = ::LoadSampleData(L"../../SampleData/CRX/DataFormat.xml", L"../../SampleData/CRX/crx.csv");
#endif
	printf("���͐M���F%d\n", pSampleData->GetDataStruct(L"input").GetDataCount());
	printf("�o�͐M���F%d\n", pSampleData->GetDataStruct(L"output").GetDataCount());

	// ���C���[DLL�Ǘ��N���X���쐬
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

	// ���C���[�f�[�^�Ǘ��N���X���쐬
	Gravisbell::Layer::NeuralNetwork::ILayerDataManager* pLayerDataManager = Gravisbell::Layer::NeuralNetwork::CreateLayerDataManager();
	if(pLayerDataManager == NULL)
	{
		delete pLayerDLLManager;
		delete pSampleData;
		return -1;
	}


	// ���o�͐M�����C���[���쐬
#if USE_GPU
#if USE_HOST_MEMORY
	Layer::IOData::IIODataLayer* pTeachInputLayer  = Layer::IOData::CreateIODataLayerGPU_host( pSampleData->GetDataStruct(L"input") );	// ���͐M��(���t�M��)
	Layer::IOData::IIODataLayer* pTeachOutputLayer = Layer::IOData::CreateIODataLayerGPU_host( pSampleData->GetDataStruct(L"output") );	// �o�͐M��(���t�M��)
	Layer::IOData::IIODataLayer* pSampleInputLayer  = Layer::IOData::CreateIODataLayerGPU_host( pSampleData->GetDataStruct(L"input") );	// ���͐M��(�m�F�p�T���v���f�[�^)
	Layer::IOData::IIODataLayer* pSampleOutputLayer = Layer::IOData::CreateIODataLayerGPU_host( pSampleData->GetDataStruct(L"output") );	// �o�͐M��(�m�F�p�T���v���f�[�^)
#else
	Layer::IOData::IIODataLayer* pTeachInputLayer  = Layer::IOData::CreateIODataLayerGPU_device( pSampleData->GetDataStruct(L"input") );	// ���͐M��(���t�M��)
	Layer::IOData::IIODataLayer* pTeachOutputLayer = Layer::IOData::CreateIODataLayerGPU_device( pSampleData->GetDataStruct(L"output") );	// �o�͐M��(���t�M��)
	Layer::IOData::IIODataLayer* pSampleInputLayer  = Layer::IOData::CreateIODataLayerGPU_device( pSampleData->GetDataStruct(L"input") );	// ���͐M��(�m�F�p�T���v���f�[�^)
	Layer::IOData::IIODataLayer* pSampleOutputLayer = Layer::IOData::CreateIODataLayerGPU_device( pSampleData->GetDataStruct(L"output") );	// �o�͐M��(�m�F�p�T���v���f�[�^)
#endif
#else
	Layer::IOData::IIODataLayer* pTeachInputLayer  = Layer::IOData::CreateIODataLayerCPU( pSampleData->GetDataStruct(L"input") );	// ���͐M��(���t�M��)
	Layer::IOData::IIODataLayer* pTeachOutputLayer = Layer::IOData::CreateIODataLayerCPU( pSampleData->GetDataStruct(L"output") );	// �o�͐M��(���t�M��)
	Layer::IOData::IIODataLayer* pSampleInputLayer  = Layer::IOData::CreateIODataLayerCPU( pSampleData->GetDataStruct(L"input") );	// ���͐M��(�m�F�p�T���v���f�[�^)
	Layer::IOData::IIODataLayer* pSampleOutputLayer = Layer::IOData::CreateIODataLayerCPU( pSampleData->GetDataStruct(L"output") );	// �o�͐M��(�m�F�p�T���v���f�[�^)
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
	printf("�P���f�[�^  �F%d\n", pTeachInputLayer->GetDataCount());
	printf("�e�X�g�f�[�^�F%d\n", pSampleInputLayer->GetDataCount());

	// �j���[�����l�b�g���[�N�N���X���쐬
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



	// �J�n�������v��
	clock_t startTime = clock();

	// �w�K,�T���v�����s�ʎ��s
#if 0
	{
		// �j���[�����l�b�g���[�N���쐬
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

		// �w�K
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

		// �T���v���Ƃ̌덷�v�Z
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

		// �������J��
		delete pNeuralNetwork;
	}
#else
	{
		// �j���[�����l�b�g���[�N���쐬
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

		// �w�K
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


		// �������J��
		delete pNeuralNetworkSample;
		delete pNeuralNetworkLearn;
	}
#endif

	// �I���������v��
	clock_t endTime = clock();

	// �o�ߎ��Ԃ�\��
	printf("��������(ms) = %d\n", endTime - startTime);

	// �������J��
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


/** �T���v���f�[�^�̓ǂݍ��� */
Gravisbell::DataFormat::IDataFormatBase* LoadSampleData(const std::wstring& formatFilePath, const std::wstring& dataFilePath)
{
	// �t�H�[�}�b�g��ǂݍ���
	auto pDataFormat = Gravisbell::DataFormat::StringArray::CreateDataFormatFromXML(formatFilePath.c_str());
	if(pDataFormat == NULL)
		return NULL;

	// CSV�t�@�C����ǂݍ���Ńt�H�[�}�b�g�ɒǉ�
	{
		// �t�@�C���I�[�v��
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

			// ","(�J���})��؂�ŕ���
			std::vector<std::wstring> lpBuf;
			boost::split(lpBuf, szBuf, boost::is_any_of(L","));

			std::vector<const wchar_t*> lpBufPointer;
			for(auto& buf : lpBuf)
				lpBufPointer.push_back(buf.c_str());


			pDataFormat->AddDataByStringArray(&lpBufPointer[0]);
		}

		// �t�@�C���N���[�Y
		fclose(fp);
	}
	// ���K��
	pDataFormat->Normalize();

	return pDataFormat;
}

/** �j���[�����l�b�g���[�N�N���X���쐬���� */
Layer::Connect::ILayerConnectData* CreateNeuralNetwork(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
{
	using namespace Gravisbell::Utility::NeuralNetworkLayer;

	Layer::Connect::ILayerConnectData* pNeuralNetwork = Gravisbell::Utility::NeuralNetworkLayer::CreateNeuralNetwork(layerDLLManager, layerDataManager);

	if(pNeuralNetwork)
	{
		// ���͐M���𒼑O���C���[�ɐݒ�
		Gravisbell::IODataStruct inputDataStruct = i_inputDataStruct;
		Gravisbell::GUID lastLayerGUID = pNeuralNetwork->GetInputGUID();

		// 1�w��
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

		// 2�w��
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

		// 3�w��(�o�͑w)
		AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateFullyConnectLayer(layerDLLManager, layerDataManager, inputDataStruct.GetDataCount(), i_outputDataStruct.GetDataCount()));
		AddLayerToNetworkLast(
			*pNeuralNetwork,
			lastLayerGUID,
			CreateActivationLayer(layerDLLManager, layerDataManager, L"softmax_ALL_crossEntropy"));

		// �o�̓��C���[�ݒ�
		pNeuralNetwork->SetOutputLayerGUID(lastLayerGUID);
	}

	pNeuralNetwork->ChangeOptimizer(L"Adam");

	return pNeuralNetwork;
}


/** �j���[�����l�b�g���[�N�̊w�K */
Gravisbell::ErrorCode LearnNeuralNetwork(
	Layer::NeuralNetwork::INeuralNetwork* pNeuralNetwork,
	Layer::IOData::IIODataLayer* pInputLayer,
	Layer::IOData::IIODataLayer* pTeachLayer,
	const U32 BATCH_SIZE,
	const U32 LEARN_TIMES)
{
	Gravisbell::ErrorCode err;

	// �w�K�W����ݒ�
	pNeuralNetwork->SetLearnSettingData(L"LearnCoeff", 0.1f);

	// ���O���������s
	err = pNeuralNetwork->PreProcessLearn(BATCH_SIZE);
	if(err != ErrorCode::ERROR_CODE_NONE)
		return err;
	err = pInputLayer->PreProcessLearn(BATCH_SIZE);
	if(err != ErrorCode::ERROR_CODE_NONE)
		return err;
	err = pTeachLayer->PreProcessLearn(BATCH_SIZE);
	if(err != ErrorCode::ERROR_CODE_NONE)
		return err;


	// �o�b�`No�����N���X���쐬
	Gravisbell::Common::IBatchDataNoListGenerator* pBatchDataNoListGenerator = Gravisbell::Common::CreateBatchDataNoListGenerator();
	err = pBatchDataNoListGenerator->PreProcess(pInputLayer->GetDataCount(), BATCH_SIZE);
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

	// �w�K�����s
	for(U32 learnTime=0; learnTime<LEARN_TIMES; learnTime++)
	{
//		printf("%4d�� ", learnTime);
		printf("%4d,",learnTime);

		// �w�K���[�v�擪����
		pBatchDataNoListGenerator->PreProcessLearnLoop();
		pInputLayer->PreProcessLearnLoop(*pLearnSetting);
		pTeachLayer->PreProcessLearnLoop(*pLearnSetting);
		pNeuralNetwork->PreProcessLearnLoop(*pLearnSetting);

		// �o�b�`�P�ʂŏ���
		for(U32 batchNum=0; batchNum<pBatchDataNoListGenerator->GetBatchDataNoListCount(); batchNum++)
		{
			// �f�[�^�؂�ւ�
			pInputLayer->SetBatchDataNoList(pBatchDataNoListGenerator->GetBatchDataNoListByNum(batchNum));
			pTeachLayer->SetBatchDataNoList(pBatchDataNoListGenerator->GetBatchDataNoListByNum(batchNum));

			// ���Z
			pNeuralNetwork->Calculate(pInputLayer->GetOutputBuffer());

			// �덷�v�Z
			// ���t�M���Ƃ̌덷�v�Z
			pTeachLayer->CalculateLearnError(pNeuralNetwork->GetOutputBuffer());

			// �w�K
			pNeuralNetwork->Training(NULL, pTeachLayer->GetDInputBuffer());
		}

		// �덷�\��
		{
			F32 errorMax, errorAve, errorAve2, errorCrossEntoropy;
			pTeachLayer->GetCalculateErrorValue(errorMax, errorAve, errorAve2, errorCrossEntoropy);
//			printf("min=%.3f, max=%.3f, ave=%.3f, ave2=%.3f\n", errorMin, errorMax, errorAve, errorAve2);
			printf("%.3f,%.3f,%.3f,%.3f\n", errorMax, errorAve, errorAve2, errorCrossEntoropy); 
		}
	}

	// �������J��
	delete pLearnSetting;
	delete pBatchDataNoListGenerator;

	return ErrorCode::ERROR_CODE_NONE;
}

/** �T���v���f�[�^�Ƃ̌덷�v�� */
Gravisbell::ErrorCode CalculateSampleError(
	Layer::NeuralNetwork::INeuralNetwork* pNeuralNetwork,
	Layer::IOData::IIODataLayer* pInputLayer,
	Layer::IOData::IIODataLayer* pTeachLayer)
{
	Gravisbell::ErrorCode err;

	// ���O���������s
	err = pNeuralNetwork->PreProcessCalculate(1);
	if(err != ErrorCode::ERROR_CODE_NONE)
		return err;
	err = pInputLayer->PreProcessCalculate(1);
	if(err != ErrorCode::ERROR_CODE_NONE)
		return err;
	err = pTeachLayer->PreProcessCalculate(1);
	if(err != ErrorCode::ERROR_CODE_NONE)
		return err;

	// ���Z���[�v�擪����
	pInputLayer->PreProcessCalculateLoop();
	pTeachLayer->PreProcessCalculateLoop();
	pNeuralNetwork->PreProcessCalculateLoop();

	// �o�b�`�P�ʂŏ���
	for(U32 dataNum=0; dataNum<pInputLayer->GetDataCount(); dataNum++)
	{
		// �f�[�^�؂�ւ�
		pInputLayer->SetBatchDataNoList(&dataNum);
		pTeachLayer->SetBatchDataNoList(&dataNum);

		// ���Z
		pNeuralNetwork->Calculate(pInputLayer->GetOutputBuffer());

		// �덷�v�Z
		pTeachLayer->CalculateLearnError(pNeuralNetwork->GetOutputBuffer());
	}

	// �덷�\��
	printf("\n�T���v���덷\n");
	{
		F32 errorMax, errorAve, errorAve2, errorCrossEntoropy;
		pTeachLayer->GetCalculateErrorValue(errorMax, errorAve, errorAve2, errorCrossEntoropy);
		printf("max=%.3f, ave=%.3f, ave2=%.3f\n", errorMax, errorAve, errorAve2);
	}

	return ErrorCode::ERROR_CODE_NONE;
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
	pNeuralNetworkLearn->SetLearnSettingData(L"LearnCoeff", 0.01f);

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
//			pBatchDataNoListGenerator->PreProcessLearnLoop();
			pTeachInputLayer->PreProcessLearnLoop(*pLearnSetting);
			pTeachOutputLayer->PreProcessLearnLoop(*pLearnSetting);
			pNeuralNetworkLearn->PreProcessLearnLoop(*pLearnSetting);

			// �w�K����
			// �o�b�`�P�ʂŏ���
			for(U32 batchNum=0; batchNum<pBatchDataNoListGenerator->GetBatchDataNoListCount(); batchNum++)
			{
				// �f�[�^�؂�ւ�
				pTeachInputLayer->SetBatchDataNoList(pBatchDataNoListGenerator->GetBatchDataNoListByNum(batchNum));
				pTeachOutputLayer->SetBatchDataNoList(pBatchDataNoListGenerator->GetBatchDataNoListByNum(batchNum));

				// ���Z
				pNeuralNetworkLearn->Calculate(pTeachInputLayer->GetOutputBuffer());

				// �덷�v�Z
				// ���t�M���Ƃ̌덷�v�Z
				pTeachOutputLayer->CalculateLearnError(pNeuralNetworkLearn->GetOutputBuffer());

				// �w�K
				pNeuralNetworkLearn->Training(NULL, pTeachOutputLayer->GetDInputBuffer());


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
			F32 errorMax, errorAve, errorAve2, errorCrossEntoropy;
			pTeachOutputLayer->GetCalculateErrorValue(errorMax, errorAve, errorAve2, errorCrossEntoropy);
//			printf("�w�K�Fmax=%.3f, ave=%.3f, ave2=%.3f, entropy=%.3f", errorMax, errorAve, errorAve2, errorCrossEntoropy);
			printf("%.3f,%.3f,%.3f,%.3f,",  errorMax, errorAve2, errorCrossEntoropy, (F32)correctCount_learn / (pBatchDataNoListGenerator->GetBatchDataNoListCount() * BATCH_SIZE)); 
		}
//		printf(" : ");
		{
			F32 errorMax, errorAve, errorAve2, errorCrossEntoropy;
			pSampleOutputLayer->GetCalculateErrorValue(errorMax, errorAve, errorAve2, errorCrossEntoropy);
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