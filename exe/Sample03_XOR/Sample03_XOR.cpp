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


#include"Library/DataFormat/DataFormatStringArray/DataFormat.h"
#include"Library/NeuralNetwork/LayerDLLManager/LayerDLLManager.h"
#include"Library/Common/BatchDataNoListGenerator/BatchDataNoListGenerator.h"
#include"Layer/IOData/IODataLayer/IODataLayer.h"
#include"Layer/NeuralNetwork/INNLayerData.h"
#include"Layer/NeuralNetwork/INNLayerConnectData.h"
#include"Layer/NeuralNetwork/INeuralNetwork.h"

using namespace Gravisbell;

/** �T���v���f�[�^�̓ǂݍ��� */
DataFormat::IDataFormatBase* LoadSampleData(const std::wstring& formatFilePath, const std::wstring& dataFilePath);
/** ���C���[DLL�Ǘ��N���X�̍쐬 */
Layer::NeuralNetwork::ILayerDLLManager* CreateLayerDLLManager(void);

/** ���C���[�f�[�^���쐬 */
Layer::NeuralNetwork::INNLayerData* CreateLayerData(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, U32 neuronCount, U32 inputDataCount, const std::wstring activationType);
Layer::NeuralNetwork::INNLayerData* CreateHiddenLayerData(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, U32 neuronCount, U32 inputDataCount);
Layer::NeuralNetwork::INNLayerData* CreateOutputLayerData(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, U32 neuronCount, U32 inputDataCount);

/** �j���[�����l�b�g���[�N�N���X���쐬���� */
Layer::NeuralNetwork::INNLayerConnectData* CreateNeuralNetwork(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, U32 inputDataCount);

/** �j���[�����l�b�g���[�N�N���X�Ƀ��C���[��ǉ����� */
Gravisbell::ErrorCode CreateNeuralNetworkLayerConnect(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager,
	Layer::NeuralNetwork::INNLayerConnectData* pNeuralNetworkData,
	U32 outputDataCount,
	std::list<Layer::ILayerData*>& lppLayerData);

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

//	srand(12345);

	std::list<Layer::ILayerData*> lppLayerData;	// ���C���[�f�[�^�̈ꗗ

	// �T���v���f�[�^�̓ǂݍ���
	Gravisbell::DataFormat::IDataFormatBase* pTeachData  = ::LoadSampleData(L"DataFormat.xml", L"../../SampleData/XOR/XOR.csv");
	printf("���͐M���F%d\n", pTeachData->GetDataStruct(L"input").GetDataCount());
	printf("�o�͐M���F%d\n", pTeachData->GetDataStruct(L"output").GetDataCount());

	// ���C���[DLL�Ǘ��N���X���쐬
	Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager = ::CreateLayerDLLManager();
	if(pLayerDLLManager == NULL)
	{
		delete pTeachData;
		return -1;
	}

	// ���o�͐M�����C���[���쐬
	Layer::IOData::IIODataLayer* pTeachInputLayer  = Layer::IOData::CreateIODataLayerCPU( pTeachData->GetDataStruct(L"input") );	// ���͐M��(���t�M��)
	Layer::IOData::IIODataLayer* pTeachOutputLayer = Layer::IOData::CreateIODataLayerCPU( pTeachData->GetDataStruct(L"output") );	// �o�͐M��(���t�M��)
	for(U32 dataNum=0; dataNum<(U32)(pTeachData->GetDataCount()); dataNum++)
	{
		pTeachInputLayer->AddData(pTeachData->GetDataByNum(dataNum, L"input"));
		pTeachOutputLayer->AddData(pTeachData->GetDataByNum(dataNum, L"output"));

		printf("INPUT : %.3f, %.3f - OUTPUT : %.3f, %.3f, %.3f\n",
			pTeachData->GetDataByNum(dataNum, L"input")[0],  pTeachData->GetDataByNum(dataNum, L"input")[1],
			pTeachData->GetDataByNum(dataNum, L"output")[0], pTeachData->GetDataByNum(dataNum, L"output")[1], pTeachData->GetDataByNum(dataNum, L"output")[2]);
	}
	printf("�P���f�[�^  �F%d\n", pTeachInputLayer->GetDataCount());

	// �j���[�����l�b�g���[�N�N���X���쐬
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

	// �j���[�����l�b�g���[�N�̐ڑ����쐬
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

	// �w�K,�T���v�����s�ʎ��s
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
			for(auto pLayerData : lppLayerData)
				delete pLayerData;
			delete pTeachInputLayer;
			delete pTeachOutputLayer;
			delete pLayerDLLManager;
			delete pTeachData;
			return -1;
		}

		// �w�K
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

		// �������J��
		delete pNeuralNetwork;
	}
	
	// �������J��
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
/** ���C���[DLL�Ǘ��N���X�̍쐬 */
Layer::NeuralNetwork::ILayerDLLManager* CreateLayerDLLManager(void)
{
	// DLL�Ǘ��N���X���쐬
	Layer::NeuralNetwork::ILayerDLLManager* pDLLManager = Layer::NeuralNetwork::CreateLayerDLLManagerCPU();

	// DLL�̓ǂݍ���.
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


/** ���C���[�f�[�^���쐬 */
Layer::NeuralNetwork::INNLayerData* CreateLayerData(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, U32 neuronCount, U32 inputDataCount, const std::wstring activationType)
{
	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(Gravisbell::GUID(0xbeba34ec, 0xc30c, 0x4565, 0x93, 0x86, 0x56, 0x08, 0x89, 0x81, 0xd2, 0xd7));
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;
	// �j���[������
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"NeuronCount"));
		pItem->SetValue(neuronCount);
	}
	// �������֐����
	{
		SettingData::Standard::IItem_Enum* pItem = dynamic_cast<SettingData::Standard::IItem_Enum*>(pConfig->GetItemByID(L"ActivationType"));
		pItem->SetValue(activationType.c_str());
//		pItem->SetValue(L"ReLU");
//		pItem->SetValue(L"sigmoid");
	}

	// ���C���[�̍쐬
	Layer::NeuralNetwork::INNLayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, IODataStruct(inputDataCount));
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
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

/** �j���[�����l�b�g���[�N�N���X���쐬���� */
Layer::NeuralNetwork::INNLayerConnectData* CreateNeuralNetwork(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, U32 inputDataCount)
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
	Layer::NeuralNetwork::INNLayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, IODataStruct(inputDataCount));
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	// �L���X�g
	Layer::NeuralNetwork::INNLayerConnectData* pNeuralNetwork = dynamic_cast<Layer::NeuralNetwork::INNLayerConnectData*>(pLayer);
	if(pNeuralNetwork == NULL)
	{
		delete pLayer;
		return NULL;
	}

	return pNeuralNetwork;
}

/** �j���[�����l�b�g���[�N�N���X�Ƀ��C���[��ǉ����� */
Gravisbell::ErrorCode CreateNeuralNetworkLayerConnect(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager,
	Layer::NeuralNetwork::INNLayerConnectData* pNeuralNetworkData,
	U32 outputDataCount,
	std::list<Layer::ILayerData*>& lppLayerData)
{
	// ���͐M���𒼑O���C���[�ɐݒ�
	Gravisbell::IODataStruct inputDataStruct = pNeuralNetworkData->GetInputDataStruct();
	Gravisbell::GUID lastLayerGUID = pNeuralNetworkData->GetInputGUID();

	// 1�w��
	{
		// GUID����
		Gravisbell::GUID guid = boost::uuids::random_generator()().data;

		// ���C���[�f�[�^���쐬
		Layer::NeuralNetwork::INNLayerData* pLayerData = CreateHiddenLayerData(layerDLLManager, 128, inputDataStruct.GetDataCount());
		if(pLayerData == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
		lppLayerData.push_back(pLayerData);
		pNeuralNetworkData->AddLayer(guid, pLayerData);

		// �ڑ�
		pNeuralNetworkData->AddInputLayerToLayer(guid, lastLayerGUID);

		// ���݃��C���[�𒼑O���C���[�ɕύX
		inputDataStruct = pLayerData->GetOutputDataStruct();
		lastLayerGUID = guid;
	}

	// 2�w��
	{
		// GUID����
		Gravisbell::GUID guid = boost::uuids::random_generator()().data;

		// ���C���[�f�[�^���쐬
		Layer::NeuralNetwork::INNLayerData* pLayerData = CreateHiddenLayerData(layerDLLManager, 128, inputDataStruct.GetDataCount());
		if(pLayerData == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
		lppLayerData.push_back(pLayerData);
		pNeuralNetworkData->AddLayer(guid, pLayerData);

		// �ڑ�
		pNeuralNetworkData->AddInputLayerToLayer(guid, lastLayerGUID);

		// ���݃��C���[�𒼑O���C���[�ɕύX
		inputDataStruct = pLayerData->GetOutputDataStruct();
		lastLayerGUID = guid;
	}

	//// 3�w��
	//{
	//	// GUID����
	//	Gravisbell::GUID guid = boost::uuids::random_generator()().data;

	//	// ���C���[�f�[�^���쐬
	//	Layer::NeuralNetwork::INNLayerData* pLayerData = CreateHiddenLayerData(layerDLLManager, 128, inputDataStruct.GetDataCount());
	//	if(pLayerData == NULL)
	//		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	//	lppLayerData.push_back(pLayerData);
	//	pNeuralNetworkData->AddLayer(guid, pLayerData);

	//	// �ڑ�
	//	pNeuralNetworkData->AddInputLayerToLayer(guid, lastLayerGUID);

	//	// ���݃��C���[�𒼑O���C���[�ɕύX
	//	inputDataStruct = pLayerData->GetOutputDataStruct();
	//	lastLayerGUID = guid;
	//}

	// 4�w��(�o�͑w)
	{
		// GUID����
		Gravisbell::GUID guid = boost::uuids::random_generator()().data;

		// ���C���[�f�[�^���쐬
		Layer::NeuralNetwork::INNLayerData* pLayerData = CreateOutputLayerData(layerDLLManager, outputDataCount, inputDataStruct.GetDataCount());
		if(pLayerData == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
		lppLayerData.push_back(pLayerData);
		pNeuralNetworkData->AddLayer(guid, pLayerData);

		// �ڑ�
		pNeuralNetworkData->AddInputLayerToLayer(guid, lastLayerGUID);

		// ���݃��C���[�𒼑O���C���[�ɕύX
		inputDataStruct = pLayerData->GetOutputDataStruct();
		lastLayerGUID = guid;

		// �o�̓��C���[�ɐݒ�
		pNeuralNetworkData->SetOutputLayerGUID(guid);
	}

	return ErrorCode::ERROR_CODE_NONE;
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
		printf("%4d�� ", learnTime);

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
			pNeuralNetwork->CalculateLearnError(pTeachLayer->GetDInputBuffer());

			// �덷�𔽉f
			pNeuralNetwork->ReflectionLearnError();
		}

		// �덷�\��
		{
			F32 errorMin, errorMax, errorAve, errorAve2, errorCrossEntoropy;
			pTeachLayer->GetCalculateErrorValue(errorMin, errorMax, errorAve, errorAve2, errorCrossEntoropy);
			printf("min=%.3f, max=%.3f, ave=%.3f, ave2=%.3f\n", errorMin, errorMax, errorAve, errorAve2);
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
		F32 errorMin, errorMax, errorAve, errorAve2, errorCrossEntoropy;
		pTeachLayer->GetCalculateErrorValue(errorMin, errorMax, errorAve, errorAve2, errorCrossEntoropy);
		printf("min=%.3f, max=%.3f, ave=%.3f, ave2=%.3f\n", errorMin, errorMax, errorAve, errorAve2);
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
	pNeuralNetworkLearn->SetLearnSettingData(L"LearnCoeff", 0.1f);

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

	// �w�K�����s
	for(U32 learnTime=0; learnTime<LEARN_TIMES; learnTime++)
	{
//		printf("%4d�� ", learnTime);
		printf("%4d,", learnTime);

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
			}
		}

		// �덷�\��
		{
			F32 errorMin, errorMax, errorAve, errorAve2, errorCrossEntoropy;
			pTeachOutputLayer->GetCalculateErrorValue(errorMin, errorMax, errorAve, errorAve2, errorCrossEntoropy);
//			printf("�w�K�Fmin=%.3f, max=%.3f, ave=%.3f, ave2=%.3f", errorMin, errorMax, errorAve, errorAve2);
			printf("%.3f,%.3f,%.3f,%.3f,", errorMin, errorMax, errorAve, errorAve2);
		}
//		printf(" : ");
		{
			F32 errorMin, errorMax, errorAve, errorAve2, errorCrossEntoropy;
			pSampleOutputLayer->GetCalculateErrorValue(errorMin, errorMax, errorAve, errorAve2, errorCrossEntoropy);
//			printf("���s�Fmin=%.3f, max=%.3f, ave=%.3f, ave2=%.3f", errorMin, errorMax, errorAve, errorAve2);
			printf("%.3f,%.3f,%.3f,%.3f,", errorMin, errorMax, errorAve, errorAve2);
		}
		printf("\n");
	}

	// �������J��
	delete pLearnSetting;
	delete pBatchDataNoListGenerator;

	return ErrorCode::ERROR_CODE_NONE;
}