// ExeSample.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
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

/** CPU制御レイヤーを作成する */
CustomDeepNNLibrary::INNLayer* CreateLayerCPU(const CustomDeepNNLibrary::INNLayerDLL* pLayerDLL, unsigned int neuronCount);

/** ニューラルネットワークテスト.
	入力層1
	中間層1-1
	出力層1
	の標準4層NN. */
void NNTest_IN1_1_1_O1(CustomDeepNNLibrary::INNLayerDLLManager& dllManager, const std::list<std::vector<float>>& lppInputA, const std::list<std::vector<float>>& lppTeachA);


int _tmain(int argc, _TCHAR* argv[])
{
#ifdef _DEBUG
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

	// DLL管理クラスを作成
	CustomDeepNNLibrary::INNLayerDLLManager* pDLLManager = CustomDeepNNLibrary::CreateLayerDLLManager();

	// DLLの読み込み
	if(pDLLManager->ReadLayerDLL(L"NNLayer_Feedforward.dll") < 0)
	{
		delete pDLLManager;
		return -1;
	}
	auto pLayerDLL = pDLLManager->GetLayerDLLByNum(0);


	// 入出力信号を作成する
	std::list<std::vector<float>> lppInputA;	// 入力A
	std::list<std::vector<float>> lppInputB;	// 入力B
	std::list<std::vector<float>> lppInputAB;	// 入力A + 入力B
	std::list<std::vector<float>> lppTeachA;	// 出力A
	std::list<std::vector<float>> lppTeachB;	// 出力B
	std::list<std::vector<float>> lppTeachAB;	// 出力A + 出力B
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

		int outputA = (inputA & inputB);	// AND演算
		int outputB = (inputA ^ inputB);	// XOR演算

		// bitにして格納
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


	//// 設定の作成
	//CustomDeepNNLibrary::INNLayerConfig* pConfig = pLayerDLL->CreateLayerConfig();
	//if(pConfig == NULL)
	//{
	//	delete pDLLManager;

	//	return -1;
	//}
	//CustomDeepNNLibrary::INNLayerConfigItem_Int* pItem = (CustomDeepNNLibrary::INNLayerConfigItem_Int*)pConfig->GetItemByNum(0);
	//pItem->SetValue(500);

	//// バッファに保存
	//std::vector<BYTE> lpBuffer(pConfig->GetUseBufferByteCount());
	//pConfig->WriteToBuffer(&lpBuffer[0]);


	//// 新しい設定をバッファから作成
	//int useBufferSize = 0;
	//CustomDeepNNLibrary::INNLayerConfig* pConfigRead = pLayerDLL->CreateLayerConfigFromBuffer(&lpBuffer[0], lpBuffer.size(), useBufferSize);


	//// 設定の削除
	//delete pConfig;
	//delete pConfigRead;

	// 動作テスト
	NNTest_IN1_1_1_O1(*pDLLManager, lppInputAB, lppTeachAB);

	// 管理クラスを削除
	delete pDLLManager;

	return 0;
}


/** CPU制御レイヤーを作成する */
CustomDeepNNLibrary::INNLayer* CreateLayerCPU(const CustomDeepNNLibrary::INNLayerDLL* pLayerDLL, unsigned int neuronCount)
{
	// 設定の作成
	CustomDeepNNLibrary::INNLayerConfig* pConfig = pLayerDLL->CreateLayerConfig();
	if(pConfig == NULL)
		return NULL;
	CustomDeepNNLibrary::INNLayerConfigItem_Int* pItem = (CustomDeepNNLibrary::INNLayerConfigItem_Int*)pConfig->GetItemByNum(0);
	pItem->SetValue(neuronCount);

	// レイヤーの作成
	CustomDeepNNLibrary::INNLayer* pLayer = pLayerDLL->CreateLayerCPU();
	if(pLayer->SetLayerConfig(*pConfig) != CustomDeepNNLibrary::LAYER_ERROR_NONE)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}



/** ニューラルネットワークテスト.
	入力層1
	中間層1-1
	出力層1
	の標準4層NN. */
void NNTest_IN1_1_1_O1(CustomDeepNNLibrary::INNLayerDLLManager& dllManager, const std::list<std::vector<float>>& lppInputA, const std::list<std::vector<float>>& lppTeachA)
{
	// レイヤーDLLの取得
	auto pLayerDLL = dllManager.GetLayerDLLByNum(0);
	if(pLayerDLL == NULL)
	{
		return;
	}

	// 全レイヤーリスト
	std::list<CustomDeepNNLibrary::ILayerBase*> lpLayer;

	// 入出力レイヤーを作成
	CustomDeepNNLibrary::IIODataLayer* pInputLayerA  = CreateIODataLayerCPU( CustomDeepNNLibrary::IODataStruct(8) );	// 出力A
	lpLayer.push_back(pInputLayerA);
	// 学習データレイヤーを作成
	CustomDeepNNLibrary::IIODataLayer* pTeachLayerA = CreateIODataLayerCPU( CustomDeepNNLibrary::IODataStruct(8) );	// 出力A

	// 入出力レイヤーにデータを格納
	for(auto& lpData : lppInputA)
		pInputLayerA->AddData(&lpData[0]);
	for(auto& lpData : lppTeachA)
		pTeachLayerA->AddData(&lpData[0]);


	// 中間層レイヤーを作成
	std::vector<CustomDeepNNLibrary::INNLayer*> lpCalcLayer;

	// 中間層1層目(合計2層目)を作成
	{
		auto pLayer = CreateLayerCPU(pLayerDLL, 20);
		if(pLayer == NULL)
		{
			for(auto pLayer : lpLayer)
				delete pLayer;
			delete pTeachLayerA;
			return;
		}

		// 入力レイヤーを設定
		pLayer->AddInputFromLayer(pInputLayerA);

		// レイヤーを登録
		lpCalcLayer.push_back(pLayer);
		lpLayer.push_back(pLayer);
	}
	
	// 中間層2層目(合計3層目)を作成
	{
		auto pLayer = CreateLayerCPU(pLayerDLL, 20);
		if(pLayer == NULL)
		{
			for(auto pLayer : lpLayer)
				delete pLayer;
			delete pTeachLayerA;
			return;
		}

		// 入力レイヤーを設定
		pLayer->AddInputFromLayer(lpCalcLayer[lpCalcLayer.size()-1]);

		// レイヤーを登録
		lpCalcLayer.push_back(pLayer);
		lpLayer.push_back(pLayer);
	}

	// 出力層(合計4層目)を作成
	auto pOutputLayer = CreateLayerCPU(pLayerDLL, pTeachLayerA->GetBufferCount());
	{
		if(pOutputLayer == NULL)
		{
			for(auto pLayer : lpLayer)
				delete pLayer;
			delete pTeachLayerA;
			return;
		}

		// 入力レイヤーを設定
		pOutputLayer->AddInputFromLayer(lpCalcLayer[lpCalcLayer.size()-1]);

		// レイヤーを登録
		lpCalcLayer.push_back(pOutputLayer);
		lpLayer.push_back(pOutputLayer);
	}


	// 学習データレイヤーを登録
	lpLayer.push_back(pTeachLayerA);
	lpCalcLayer[lpCalcLayer.size()-1]->AddOutputToLayer(pTeachLayerA);

	// レイヤーを初期化する
	for(auto pLayer : lpCalcLayer)
	{
		if(pLayer->Initialize() != CustomDeepNNLibrary::LAYER_ERROR_NONE)
		{
			// レイヤーを削除
			for(auto pLayer : lpLayer)
				delete pLayer;
			return;
		}
	}

	// 事前処理を実行する
	for(auto pLayer : lpLayer)
	{
		if(pLayer->PreCalculate() != CustomDeepNNLibrary::LAYER_ERROR_NONE)
		{
			// レイヤーを削除
			for(auto pLayer : lpLayer)
				delete pLayer;
			return;
		}
	}

	// 演算処理を実行する
	for(unsigned int calcNum=0; calcNum<200; calcNum++)
	{
		for(unsigned int dataNum=0; dataNum<pInputLayerA->GetDataCount(); dataNum++)
		{
			// データ切り替え
			pInputLayerA->ChangeUseDataByNum(dataNum);
			pTeachLayerA->ChangeUseDataByNum(dataNum);

			// 演算
			for(auto pLayer : lpCalcLayer)
				pLayer->Calculate();

			// 誤差計算
			// 誤差計算は逆順
			pTeachLayerA->CalculateLearnError();
			auto it = lpCalcLayer.rbegin();
			while(it != lpCalcLayer.rend())
			{
				(*it)->CalculateLearnError();
				it++;
			}

			// 誤差を反映
			for(auto pLayer : lpCalcLayer)
				pLayer->ReflectionLearnError();
		}
	}

	// サンプルを実行
	for(unsigned int dataNum=0; dataNum<pInputLayerA->GetDataCount(); dataNum++)
	{
		// データ切り替え
		pInputLayerA->ChangeUseDataByNum(dataNum);
		pTeachLayerA->ChangeUseDataByNum(dataNum);

		// 演算
		for(auto pLayer : lpCalcLayer)
			pLayer->Calculate();


		printf("No.%d\n", dataNum);
		// 入力を表示
		printf("入力A ");
		for(unsigned int i=0; i<pInputLayerA->GetOutputBufferCount()/2; i++)
			printf("%d", pInputLayerA->GetOutputBuffer()[i]>0.5f ? 1: 0);
		printf("\n");
		printf("入力B ");
		for(unsigned int i=0; i<pInputLayerA->GetOutputBufferCount()/2; i++)
			printf("%d", pInputLayerA->GetOutputBuffer()[i + pInputLayerA->GetOutputBufferCount()/2] > 0.5f ? 1 : 0);
		printf("\n");
		// 出力を表示
		printf("教師A ");
		for(unsigned int i=0; i<pTeachLayerA->GetOutputBufferCount()/2; i++)
			printf("%d", pTeachLayerA->GetOutputBuffer()[i] > 0.5f ? 1 : 0);
		printf("\n");
		printf("出力A ");
		for(unsigned int i=0; i<pOutputLayer->GetOutputBufferCount()/2; i++)
			printf("%d", pOutputLayer->GetOutputBuffer()[i] > 0.5f ? 1 : 0);
		printf("\n");
		printf("教師B ");
		for(unsigned int i=0; i<pTeachLayerA->GetOutputBufferCount()/2; i++)
			printf("%d", pTeachLayerA->GetOutputBuffer()[i + pTeachLayerA->GetOutputBufferCount()/2] > 0.5f ? 1 : 0);
		printf("\n");
		printf("出力B ");
		for(unsigned int i=0; i<pOutputLayer->GetOutputBufferCount()/2; i++)
			printf("%d", pOutputLayer->GetOutputBuffer()[i + pOutputLayer->GetOutputBufferCount()/2] > 0.5f ? 1 : 0);
		printf("\n");

		printf("\n");
	}

	// レイヤーを削除
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