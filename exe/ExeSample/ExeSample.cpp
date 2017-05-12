// ExeSample.cpp : �R���\�[�� �A�v���P�[�V�����̃G���g�� �|�C���g���`���܂��B
//

#include "stdafx.h"

#include<vector>
#include<list>
#include<boost/uuid/uuid_generators.hpp>

#include"SettingData/Standard/IData.h"
#include"Layer/IOData/IIODataLayer.h"

#include"../Library/NeuralNetwork/LayerDLLManager/LayerDLLManager.h"
#include"../Library/Common/BatchDataNoListGenerator/BatchDataNoListGenerator.h"
#include"../Layer/IOData/IODataLayer/IODataLayer.h"
#include"Layer/NeuralNetwork/INNLayerData.h"

#include<Windows.h>

using namespace Gravisbell;

/** CPU���䃌�C���[���쐬���� */
Layer::NeuralNetwork::INNLayerData* CreateHiddenLayerCPU(const Layer::NeuralNetwork::ILayerDLL* pLayerDLL, U32 neuronCount, U32 inputDataCount);
/** CPU���䃌�C���[���쐬���� */
Layer::NeuralNetwork::INNLayerData* CreateOutputLayerCPU(const Layer::NeuralNetwork::ILayerDLL* pLayerDLL, U32 neuronCount, U32 inputDataCount);

/** �j���[�����l�b�g���[�N�e�X�g.
	���͑w1
	���ԑw1-1
	�o�͑w1
	�̕W��4�wNN. */
void NNTest_IN1_1_1_O1(Layer::NeuralNetwork::ILayerDLLManager& dllManager, const std::list<std::vector<float>>& lppInputA, const std::list<std::vector<float>>& lppTeachA);


int _tmain(int argc, _TCHAR* argv[])
{
#ifdef _DEBUG
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

	// DLL�Ǘ��N���X���쐬
	Layer::NeuralNetwork::ILayerDLLManager* pDLLManager = Layer::NeuralNetwork::CreateLayerDLLManagerCPU();

	// DLL�̓ǂݍ���
#ifdef _DEBUG
	if(pDLLManager->ReadLayerDLL(L"../../Debug/Gravisbell.Layer.NeuralNetwork.FullyConnect_Activation.dll") != Gravisbell::ErrorCode::ERROR_CODE_NONE)
#else
	if(pDLLManager->ReadLayerDLL(L"../../Release/Gravisbell.Layer.NeuralNetwork.FullyConnect_Activation.dll") != Gravisbell::ErrorCode::ERROR_CODE_NONE)
#endif
	{
		delete pDLLManager;
		return -1;
	}
	auto pLayerDLL = pDLLManager->GetLayerDLLByNum(0);


	// ���o�͐M�����쐬����
	std::list<std::vector<float>> lppInputA;	// ����A
	std::list<std::vector<float>> lppInputB;	// ����B
	std::list<std::vector<float>> lppInputAB;	// ����A + ����B
	std::list<std::vector<float>> lppTeachA;	// �o��A
	std::list<std::vector<float>> lppTeachB;	// �o��B
	std::list<std::vector<float>> lppTeachAB;	// �o��A + �o��B
	for(unsigned int i=0; i<256; i++)
	{
		std::vector<float> lpInputA(4);
		std::vector<float> lpInputB(4);
		std::vector<float> lpInputAB(8);

		std::vector<float> lpTeachA(4);
		std::vector<float> lpTeachB(4);
		std::vector<float> lpTeachAB(8);

		int inputA = (i & 0xF0) >> 4;
		int inputB = (i & 0x0F) >> 0;

		int outputA = (inputA & inputB);	// AND���Z
		int outputB = (inputA ^ inputB);	// XOR���Z

		// bit�ɂ��Ċi�[
		for(int i=0; i<4; i++)
		{
			lpInputA[i] = (float)((inputA  >> i) & 0x01) * 0.90f + 0.05f;
			lpInputB[i] = (float)((inputB  >> i) & 0x01) * 0.90f + 0.05f;
			lpTeachA[i] = (float)((outputA >> i) & 0x01) * 0.90f + 0.05f;
			lpTeachB[i] = (float)((outputB >> i) & 0x01) * 0.90f + 0.05f;

			lpInputAB[i + 0] = lpInputA[i];
			lpInputAB[i + 4] = lpInputB[i];
			lpTeachAB[i + 0] = lpTeachA[i];
			lpTeachAB[i + 4] = lpTeachB[i];
		}

		lppInputA.push_back(lpInputA);
		lppInputB.push_back(lpInputB);
		lppInputAB.push_back(lpInputAB);
		lppTeachA.push_back(lpTeachA);
		lppTeachB.push_back(lpTeachB);
		lppTeachAB.push_back(lpTeachAB);
	}


	//// �ݒ�̍쐬
	//CustomDeepNNLibrary::INNLayerConfig* pConfig = pLayerDLL->CreateLayerConfig();
	//if(pConfig == NULL)
	//{
	//	delete pDLLManager;

	//	return -1;
	//}
	//CustomDeepNNLibrary::INNLayerConfigItem_Int* pItem = (CustomDeepNNLibrary::INNLayerConfigItem_Int*)pConfig->GetItemByNum(0);
	//pItem->SetValue(500);

	//// �o�b�t�@�ɕۑ�
	//std::vector<BYTE> lpBuffer(pConfig->GetUseBufferByteCount());
	//pConfig->WriteToBuffer(&lpBuffer[0]);


	//// �V�����ݒ���o�b�t�@����쐬
	//int useBufferSize = 0;
	//CustomDeepNNLibrary::INNLayerConfig* pConfigRead = pLayerDLL->CreateLayerConfigFromBuffer(&lpBuffer[0], lpBuffer.size(), useBufferSize);


	//// �ݒ�̍폜
	//delete pConfig;
	//delete pConfigRead;

	// ����e�X�g
	NNTest_IN1_1_1_O1(*pDLLManager, lppInputAB, lppTeachAB);

	// �Ǘ��N���X���폜
	delete pDLLManager;

	printf("Press any key to continue");
	getc(stdin);

	return 0;
}


/** CPU���䃌�C���[���쐬���� */
Layer::NeuralNetwork::INNLayerData* CreateHiddenLayerCPU(const Layer::NeuralNetwork::ILayerDLL* pLayerDLL, U32 neuronCount, U32 inputDataCount)
{
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
		pItem->SetValue(L"ReLU");
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
/** CPU���䃌�C���[���쐬���� */
Layer::NeuralNetwork::INNLayerData* CreateOutputLayerCPU(const Layer::NeuralNetwork::ILayerDLL* pLayerDLL, U32 neuronCount, U32 inputDataCount)
{
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
//		pItem->SetValue(L"ReLU");
		pItem->SetValue(L"sigmoid");
	}

	// ���C���[�̍쐬
	Layer::NeuralNetwork::INNLayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, IODataStruct(inputDataCount));
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}




/** �j���[�����l�b�g���[�N�e�X�g.
	���͑w1
	���ԑw1-1
	�o�͑w1
	�̕W��4�wNN. */
void NNTest_IN1_1_1_O1(Layer::NeuralNetwork::ILayerDLLManager& dllManager, const std::list<std::vector<float>>& lppInputA, const std::list<std::vector<float>>& lppTeachA)
{
	const U32 BATCH_SIZE = 32;


	// ���C���[DLL�̎擾
	auto pLayerDLL = dllManager.GetLayerDLLByNum(0);
	if(pLayerDLL == NULL)
	{
		return;
	}

	// �w�K�ݒ�f�[�^�̍쐬
	// ���o�̓��C���[
	auto pLearnSettingIO = Gravisbell::Layer::IOData::CreateLearningSetting();
	{
	}
	// ���ԑw(���͎󂯎��)
	auto pLearnSettingCalcInput = pLayerDLL->CreateLearningSetting();
	{
		// �w�K�W��
		auto pItemLearnCoeff = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Float*>(pLearnSettingCalcInput->GetItemByID(L"LearnCoeff"));
		if(pItemLearnCoeff)
			pItemLearnCoeff->SetValue(0.01f);
		// �h���b�v�A�E�g��
		auto pItemDropOut = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Float*>(pLearnSettingCalcInput->GetItemByID(L"DropOut"));
		if(pItemDropOut)
			pItemDropOut->SetValue(0.2f);
	}
	// ���ԑw
	auto pLearnSettingCalc = pLayerDLL->CreateLearningSetting();
	{
		// �w�K�W��
		auto pItemLearnCoeff = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Float*>(pLearnSettingCalc->GetItemByID(L"LearnCoeff"));
		if(pItemLearnCoeff)
			pItemLearnCoeff->SetValue(0.01f);
		// �h���b�v�A�E�g��
		auto pItemDropOut = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Float*>(pLearnSettingCalc->GetItemByID(L"DropOut"));
		if(pItemDropOut)
			pItemDropOut->SetValue(0.5f);
	}
	// �o�͑w
	auto pLearnSettingCalcOutput = pLayerDLL->CreateLearningSetting();
	{
		// �w�K�W��
		auto pItemLearnCoeff = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Float*>(pLearnSettingCalcOutput->GetItemByID(L"LearnCoeff"));
		if(pItemLearnCoeff)
			pItemLearnCoeff->SetValue(0.01f);
		// �h���b�v�A�E�g��
		auto pItemDropOut = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Float*>(pLearnSettingCalcOutput->GetItemByID(L"DropOut"));
		if(pItemDropOut)
			pItemDropOut->SetValue(0.0f);
	}

	// �S���C���[���X�g
	std::list<Layer::ILayerBase*> lpLayer;
	std::list<Layer::ILayerData*> lpLayerData;

	// ���o�̓��C���[���쐬
	Layer::IOData::IIODataLayer* pInputLayerA  = Layer::IOData::CreateIODataLayerCPU( IODataStruct(8) );	// �o��A
	lpLayer.push_back(pInputLayerA);
	// �w�K�f�[�^���C���[���쐬
	Layer::IOData::IIODataLayer* pTeachLayerA  = Layer::IOData::CreateIODataLayerCPU( IODataStruct(8) );	// �o��A

	// ���o�̓��C���[�Ƀf�[�^���i�[
	for(auto& lpData : lppInputA)
		pInputLayerA->AddData(&lpData[0]);
	for(auto& lpData : lppTeachA)
		pTeachLayerA->AddData(&lpData[0]);


	// ���ԑw���C���[���쐬
	std::vector<Layer::NeuralNetwork::INNLayer*> lpCalcLayer;

	// ���ԑw1�w��(���v2�w��)���쐬
	{
		// ���C���[�f�[�^�쐬
		auto pLayerData = CreateHiddenLayerCPU(pLayerDLL, 80, pInputLayerA->GetOutputBufferCount());
		if(pLayerData == NULL)
		{
			for(auto pLayer : lpLayer)
				delete pLayer;
			for(auto pLayerData : lpLayerData)
				delete pLayerData;
			delete pTeachLayerA;
			delete pLearnSettingIO;
			delete pLearnSettingCalc;
			return;
		}
		lpLayerData.push_back(pLayerData);

		// ���C���[�쐬
		auto pLayer = pLayerData->CreateLayer(boost::uuids::random_generator()().data);

		// ���C���[��o�^
		lpCalcLayer.push_back(pLayer);
		lpLayer.push_back(pLayer);
	}
	
	// ���ԑw2�w��(���v3�w��)���쐬
	{
		// ���C���[�f�[�^�쐬
		auto pLayerData = CreateHiddenLayerCPU(pLayerDLL, 80, (*lpCalcLayer.rbegin())->GetOutputBufferCount());
		if(pLayerData == NULL)
		{
			for(auto pLayer : lpLayer)
				delete pLayer;
			for(auto pLayerData : lpLayerData)
				delete pLayerData;
			delete pTeachLayerA;
			delete pLearnSettingIO;
			delete pLearnSettingCalc;
			return;
		}
		lpLayerData.push_back(pLayerData);

		// ���C���[�쐬
		auto pLayer = pLayerData->CreateLayer(boost::uuids::random_generator()().data);

		// ���C���[��o�^
		lpCalcLayer.push_back(pLayer);
		lpLayer.push_back(pLayer);
	}

	// �o�͑w(���v4�w��)���쐬
	Gravisbell::Layer::NeuralNetwork::INNLayer* pOutputLayer = NULL;
	{
		auto pLayerData = CreateOutputLayerCPU(pLayerDLL, pTeachLayerA->GetBufferCount(), (*lpCalcLayer.rbegin())->GetOutputBufferCount());

		if(pLayerData == NULL)
		{
			for(auto pLayer : lpLayer)
				delete pLayer;
			for(auto pLayerData : lpLayerData)
				delete pLayerData;
			delete pTeachLayerA;
			delete pLearnSettingIO;
			delete pLearnSettingCalc;
			return;
		}
		lpLayerData.push_back(pLayerData);

		// ���C���[�쐬
		pOutputLayer = pLayerData->CreateLayer(boost::uuids::random_generator()().data);

		// ���C���[��o�^
		lpCalcLayer.push_back(pOutputLayer);
		lpLayer.push_back(pOutputLayer);
	}


	// �w�K�f�[�^���C���[��o�^
	lpLayer.push_back(pTeachLayerA);


	// ���C���[������������
	for(auto pLayer : lpCalcLayer)
	{
		if(pLayer->Initialize() != ErrorCode::ERROR_CODE_NONE)
		{
			// ���C���[���폜
			for(auto pLayer : lpLayer)
				delete pLayer;
			for(auto pLayerData : lpLayerData)
				delete pLayerData;
			delete pLearnSettingIO;
			delete pLearnSettingCalc;
			return;
		}
	}

	// ���O���������s����
	for(auto pLayer : lpLayer)
	{
		if(pLayer->PreProcessLearn(BATCH_SIZE) != ErrorCode::ERROR_CODE_NONE)
		{
			// ���C���[���폜
			for(auto pLayer : lpLayer)
				delete pLayer;
			for(auto pLayerData : lpLayerData)
				delete pLayerData;
			delete pLearnSettingIO;
			delete pLearnSettingCalc;
			return;
		}
	}

	// �o�b�`No�����N���X���쐬
	Gravisbell::Common::IBatchDataNoListGenerator* pBatchDataNoListGenerator = Gravisbell::Common::CreateBatchDataNoListGenerator();
	pBatchDataNoListGenerator->PreProcess(pInputLayerA->GetDataCount(), BATCH_SIZE);

	// ���Z���������s����
	for(unsigned int calcNum=0; calcNum<20; calcNum++)
	{
		printf("%4d�� ", calcNum);

		// �w�K���[�v�擪����
		pBatchDataNoListGenerator->PreProcessLearnLoop();
		pInputLayerA->PreProcessLearnLoop(*pLearnSettingIO);
		pTeachLayerA->PreProcessLearnLoop(*pLearnSettingIO);
		lpCalcLayer[0]->PreProcessLearnLoop(*pLearnSettingCalcInput);
		for(U32 layerNum=1; layerNum<lpCalcLayer.size()-1; layerNum++)
		{
			lpCalcLayer[layerNum]->PreProcessLearnLoop(*pLearnSettingCalc);
		}
		lpCalcLayer[lpCalcLayer.size()-1]->PreProcessLearnLoop(*pLearnSettingCalcOutput);

		for(unsigned int batchNum=0; batchNum<pBatchDataNoListGenerator->GetBatchDataNoListCount(); batchNum++)
		{
			// �f�[�^�؂�ւ�
			pInputLayerA->SetBatchDataNoList(pBatchDataNoListGenerator->GetBatchDataNoListByNum(batchNum));
			pTeachLayerA->SetBatchDataNoList(pBatchDataNoListGenerator->GetBatchDataNoListByNum(batchNum));

			// ���Z
			lpCalcLayer[0]->Calculate(pInputLayerA->GetOutputBuffer());
			for(U32 layerNum=1; layerNum<lpCalcLayer.size(); layerNum++)
			{
				lpCalcLayer[layerNum]->Calculate(lpCalcLayer[layerNum-1]->GetOutputBuffer());
			}

			// �덷�v�Z
			// ���t�M���Ƃ̌덷�v�Z
			pTeachLayerA->CalculateLearnError((*lpCalcLayer.rbegin())->GetOutputBuffer());

			// �덷�v�Z�͋t��
			auto it = lpCalcLayer.rbegin();
			(*it)->Training(pTeachLayerA->GetDInputBuffer());
			auto it_last = it;
			it++;
			while(it != lpCalcLayer.rend())
			{
				(*it)->Training((*it_last)->GetDInputBuffer());
				it_last = it;
				it++;
			}
		}

		// �덷�\��
		{
			F32 errorMin, errorMax, errorAve, errorAve2, errorCrossEntoropy;
			pTeachLayerA->GetCalculateErrorValue(errorMin, errorMax, errorAve, errorAve2, errorCrossEntoropy);
			printf("min=%.3f, max=%.3f, ave=%.3f, ave2=%.3f\n", errorMin, errorMax, errorAve, errorAve2);
		}
	}

	// �o�b�`�T�C�Y��1�ɂ��ĉ��Z�p�ɐ؂�ւ�
	for(auto pLayer : lpLayer)
	{
		if(pLayer->PreProcessCalculate(1) != ErrorCode::ERROR_CODE_NONE)
		{
			// ���C���[���폜
			for(auto pLayer : lpLayer)
				delete pLayer;
			for(auto pLayerData : lpLayerData)
				delete pLayerData;
			delete pLearnSettingIO;
			delete pLearnSettingCalcInput;
			delete pLearnSettingCalc;
			return;
		}
	}

	// �v�Z���[�v�O���������s
	for(auto pLayer : lpLayer)
		pLayer->PreProcessCalculateLoop();

	// �T���v�������s
	for(unsigned int dataNum=0; dataNum<pInputLayerA->GetDataCount(); dataNum++)
	{
		// �f�[�^�؂�ւ�
		pInputLayerA->SetBatchDataNoList(&dataNum);
		pTeachLayerA->SetBatchDataNoList(&dataNum);

		// ���Z
		lpCalcLayer[0]->Calculate(pInputLayerA->GetOutputBuffer());
		for(U32 layerNum=1; layerNum<lpCalcLayer.size(); layerNum++)
		{
			lpCalcLayer[layerNum]->Calculate(lpCalcLayer[layerNum-1]->GetOutputBuffer());
		}

		// �o�͑w�̌덷�v�Z���s��
		pTeachLayerA->CalculateLearnError(pOutputLayer->GetOutputBuffer());

#if 0
		printf("No.%d\n", dataNum);
		// ���͂�\��
		printf("����A ");
		for(unsigned int i=0; i<pInputLayerA->GetOutputBufferCount()/2; i++)
			printf("%d", pInputLayerA->GetOutputBuffer()[0][i]>0.5f ? 1: 0);
		printf("\n");
		printf("����B ");
		for(unsigned int i=0; i<pInputLayerA->GetOutputBufferCount()/2; i++)
			printf("%d", pInputLayerA->GetOutputBuffer()[0][i + pInputLayerA->GetOutputBufferCount()/2] > 0.5f ? 1 : 0);
		printf("\n");
		// �o�͂�\��
		printf("���tA ");
		for(unsigned int i=0; i<pTeachLayerA->GetOutputBufferCount()/2; i++)
			printf("%d", pTeachLayerA->GetOutputBuffer()[0][i] > 0.5f ? 1 : 0);
		printf("\n");
		printf("�o��A ");
		for(unsigned int i=0; i<pOutputLayer->GetOutputBufferCount()/2; i++)
			printf("%d", pOutputLayer->GetOutputBuffer()[0][i] > 0.5f ? 1 : 0);
		printf("\n");
		printf("���tB ");
		for(unsigned int i=0; i<pTeachLayerA->GetOutputBufferCount()/2; i++)
			printf("%d", pTeachLayerA->GetOutputBuffer()[0][i + pTeachLayerA->GetOutputBufferCount()/2] > 0.5f ? 1 : 0);
		printf("\n");
		printf("�o��B ");
		for(unsigned int i=0; i<pOutputLayer->GetOutputBufferCount()/2; i++)
			printf("%d", pOutputLayer->GetOutputBuffer()[0][i + pOutputLayer->GetOutputBufferCount()/2] > 0.5f ? 1 : 0);
		printf("\n");

		printf("\n");
#endif

	}
	
	// �덷�\��
	{
		F32 errorMin, errorMax, errorAve, errorAve2, errorCrossEntoropy;
		pTeachLayerA->GetCalculateErrorValue(errorMin, errorMax, errorAve, errorAve2, errorCrossEntoropy);
		printf("min=%.3f, max=%.3f, ave=%.3f, ave2=%.3f\n", errorMin, errorMax, errorAve, errorAve2);
	}

	// ���C���[���폜
	for(auto pLayer : lpLayer)
		delete pLayer;
	for(auto pLayerData : lpLayerData)
		delete pLayerData;
	delete pLearnSettingIO;
	delete pLearnSettingCalcInput;
	delete pLearnSettingCalc;
	delete pLearnSettingCalcOutput;

	// �o�b�`No�����N���X���폜
	delete pBatchDataNoListGenerator;
}
