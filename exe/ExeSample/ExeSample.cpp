// ExeSample.cpp : �R���\�[�� �A�v���P�[�V�����̃G���g�� �|�C���g���`���܂��B
//

#include "stdafx.h"

#include<vector>
#include<list>

#include"SettingData/Standard/IData.h"
#include"NNLayerInterface\INNLayerDLL.h"
#include"NNLayerInterface\IIODataLayer.h"

#include"../Library/NNLayerDLLManager/NNLayerDLLManager.h"
#include"../NNLayer/IODataLayer/IODataLayer.h"

#include<Windows.h>

#if 0

/** CPU���䃌�C���[���쐬���� */
CustomDeepNNLibrary::INNLayer* CreateLayerCPU(const CustomDeepNNLibrary::INNLayerDLL* pLayerDLL, unsigned int neuronCount);

/** �j���[�����l�b�g���[�N�e�X�g.
	���͑w1
	���ԑw1-1
	�o�͑w1
	�̕W��4�wNN. */
void NNTest_IN1_1_1_O1(CustomDeepNNLibrary::INNLayerDLLManager& dllManager, const std::list<std::vector<float>>& lppInputA, const std::list<std::vector<float>>& lppTeachA);


int _tmain(int argc, _TCHAR* argv[])
{
#ifdef _DEBUG
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

	// DLL�Ǘ��N���X���쐬
	CustomDeepNNLibrary::INNLayerDLLManager* pDLLManager = CustomDeepNNLibrary::CreateLayerDLLManager();

	// DLL�̓ǂݍ���
	if(pDLLManager->ReadLayerDLL(L"NNLayer_Feedforward.dll") < 0)
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

	return 0;
}


/** CPU���䃌�C���[���쐬���� */
CustomDeepNNLibrary::INNLayer* CreateLayerCPU(const CustomDeepNNLibrary::INNLayerDLL* pLayerDLL, unsigned int neuronCount)
{
	// �ݒ�̍쐬
	CustomDeepNNLibrary::INNLayerConfig* pConfig = pLayerDLL->CreateLayerConfig();
	if(pConfig == NULL)
		return NULL;
	CustomDeepNNLibrary::INNLayerConfigItem_Int* pItem = (CustomDeepNNLibrary::INNLayerConfigItem_Int*)pConfig->GetItemByNum(0);
	pItem->SetValue(neuronCount);

	// ���C���[�̍쐬
	CustomDeepNNLibrary::INNLayer* pLayer = pLayerDLL->CreateLayerCPU();
	if(pLayer->SetLayerConfig(*pConfig) != CustomDeepNNLibrary::LAYER_ERROR_NONE)
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
void NNTest_IN1_1_1_O1(CustomDeepNNLibrary::INNLayerDLLManager& dllManager, const std::list<std::vector<float>>& lppInputA, const std::list<std::vector<float>>& lppTeachA)
{
	// ���C���[DLL�̎擾
	auto pLayerDLL = dllManager.GetLayerDLLByNum(0);
	if(pLayerDLL == NULL)
	{
		return;
	}

	// �S���C���[���X�g
	std::list<CustomDeepNNLibrary::ILayerBase*> lpLayer;

	// ���o�̓��C���[���쐬
	CustomDeepNNLibrary::IIODataLayer* pInputLayerA  = CreateIODataLayerCPU( CustomDeepNNLibrary::IODataStruct(8) );	// �o��A
	lpLayer.push_back(pInputLayerA);
	// �w�K�f�[�^���C���[���쐬
	CustomDeepNNLibrary::IIODataLayer* pTeachLayerA = CreateIODataLayerCPU( CustomDeepNNLibrary::IODataStruct(8) );	// �o��A

	// ���o�̓��C���[�Ƀf�[�^���i�[
	for(auto& lpData : lppInputA)
		pInputLayerA->AddData(&lpData[0]);
	for(auto& lpData : lppTeachA)
		pTeachLayerA->AddData(&lpData[0]);


	// ���ԑw���C���[���쐬
	std::vector<CustomDeepNNLibrary::INNLayer*> lpCalcLayer;

	// ���ԑw1�w��(���v2�w��)���쐬
	{
		auto pLayer = CreateLayerCPU(pLayerDLL, 20);
		if(pLayer == NULL)
		{
			for(auto pLayer : lpLayer)
				delete pLayer;
			delete pTeachLayerA;
			return;
		}

		// ���̓��C���[��ݒ�
		pLayer->AddInputFromLayer(pInputLayerA);

		// ���C���[��o�^
		lpCalcLayer.push_back(pLayer);
		lpLayer.push_back(pLayer);
	}
	
	// ���ԑw2�w��(���v3�w��)���쐬
	{
		auto pLayer = CreateLayerCPU(pLayerDLL, 20);
		if(pLayer == NULL)
		{
			for(auto pLayer : lpLayer)
				delete pLayer;
			delete pTeachLayerA;
			return;
		}

		// ���̓��C���[��ݒ�
		pLayer->AddInputFromLayer(lpCalcLayer[lpCalcLayer.size()-1]);

		// ���C���[��o�^
		lpCalcLayer.push_back(pLayer);
		lpLayer.push_back(pLayer);
	}

	// �o�͑w(���v4�w��)���쐬
	auto pOutputLayer = CreateLayerCPU(pLayerDLL, pTeachLayerA->GetBufferCount());
	{
		if(pOutputLayer == NULL)
		{
			for(auto pLayer : lpLayer)
				delete pLayer;
			delete pTeachLayerA;
			return;
		}

		// ���̓��C���[��ݒ�
		pOutputLayer->AddInputFromLayer(lpCalcLayer[lpCalcLayer.size()-1]);

		// ���C���[��o�^
		lpCalcLayer.push_back(pOutputLayer);
		lpLayer.push_back(pOutputLayer);
	}


	// �w�K�f�[�^���C���[��o�^
	lpLayer.push_back(pTeachLayerA);
	lpCalcLayer[lpCalcLayer.size()-1]->AddOutputToLayer(pTeachLayerA);

	// ���C���[������������
	for(auto pLayer : lpCalcLayer)
	{
		if(pLayer->Initialize() != CustomDeepNNLibrary::LAYER_ERROR_NONE)
		{
			// ���C���[���폜
			for(auto pLayer : lpLayer)
				delete pLayer;
			return;
		}
	}

	// ���O���������s����
	for(auto pLayer : lpLayer)
	{
		if(pLayer->PreCalculate() != CustomDeepNNLibrary::LAYER_ERROR_NONE)
		{
			// ���C���[���폜
			for(auto pLayer : lpLayer)
				delete pLayer;
			return;
		}
	}

	// ���Z���������s����
	for(unsigned int calcNum=0; calcNum<200; calcNum++)
	{
		for(unsigned int dataNum=0; dataNum<pInputLayerA->GetDataCount(); dataNum++)
		{
			// �f�[�^�؂�ւ�
			pInputLayerA->ChangeUseDataByNum(dataNum);
			pTeachLayerA->ChangeUseDataByNum(dataNum);

			// ���Z
			for(auto pLayer : lpCalcLayer)
				pLayer->Calculate();

			// �덷�v�Z
			// �덷�v�Z�͋t��
			pTeachLayerA->CalculateLearnError();
			auto it = lpCalcLayer.rbegin();
			while(it != lpCalcLayer.rend())
			{
				(*it)->CalculateLearnError();
				it++;
			}

			// �덷�𔽉f
			for(auto pLayer : lpCalcLayer)
				pLayer->ReflectionLearnError();
		}
	}

	// �T���v�������s
	for(unsigned int dataNum=0; dataNum<pInputLayerA->GetDataCount(); dataNum++)
	{
		// �f�[�^�؂�ւ�
		pInputLayerA->ChangeUseDataByNum(dataNum);
		pTeachLayerA->ChangeUseDataByNum(dataNum);

		// ���Z
		for(auto pLayer : lpCalcLayer)
			pLayer->Calculate();


		printf("No.%d\n", dataNum);
		// ���͂�\��
		printf("����A ");
		for(unsigned int i=0; i<pInputLayerA->GetOutputBufferCount()/2; i++)
			printf("%d", pInputLayerA->GetOutputBuffer()[i]>0.5f ? 1: 0);
		printf("\n");
		printf("����B ");
		for(unsigned int i=0; i<pInputLayerA->GetOutputBufferCount()/2; i++)
			printf("%d", pInputLayerA->GetOutputBuffer()[i + pInputLayerA->GetOutputBufferCount()/2] > 0.5f ? 1 : 0);
		printf("\n");
		// �o�͂�\��
		printf("���tA ");
		for(unsigned int i=0; i<pTeachLayerA->GetOutputBufferCount()/2; i++)
			printf("%d", pTeachLayerA->GetOutputBuffer()[i] > 0.5f ? 1 : 0);
		printf("\n");
		printf("�o��A ");
		for(unsigned int i=0; i<pOutputLayer->GetOutputBufferCount()/2; i++)
			printf("%d", pOutputLayer->GetOutputBuffer()[i] > 0.5f ? 1 : 0);
		printf("\n");
		printf("���tB ");
		for(unsigned int i=0; i<pTeachLayerA->GetOutputBufferCount()/2; i++)
			printf("%d", pTeachLayerA->GetOutputBuffer()[i + pTeachLayerA->GetOutputBufferCount()/2] > 0.5f ? 1 : 0);
		printf("\n");
		printf("�o��B ");
		for(unsigned int i=0; i<pOutputLayer->GetOutputBufferCount()/2; i++)
			printf("%d", pOutputLayer->GetOutputBuffer()[i + pOutputLayer->GetOutputBufferCount()/2] > 0.5f ? 1 : 0);
		printf("\n");

		printf("\n");
	}

	// ���C���[���폜
	for(auto pLayer : lpLayer)
		delete pLayer;
}

#endif



int _tmain(int argc, _TCHAR* argv[])
{
	Gravisbell::IODataStruct ioDataStruct(1, 1, 1, 1);

	Gravisbell::NeuralNetwork::IIODataLayer* pIODataLayer = ::CreateIODataLayerCPU(ioDataStruct);

	printf("Press any key to Continue\n");
	getc(stdin);

	return 0;
}