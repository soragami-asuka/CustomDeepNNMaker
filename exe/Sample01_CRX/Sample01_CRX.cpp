//=============================================
// クレジットカード認証のデータを用いた実装サンプル
// 参考URL：
// ・Dropout：ディープラーニングの火付け役、単純な方法で過学習を防ぐ
//	https://deepage.net/deep_learning/2016/10/17/deeplearning_dropout.html
//
// サンプルデータURL:
// https://archive.ics.uci.edu/ml/datasets/Credit+Approval
//  データ本体
//		Data Folder > crx.data
//  データフォーマットについて
//		Data Folder > crx.names
//=============================================


#include "stdafx.h"

#include <boost/tokenizer.hpp>
#include<boost/algorithm/string.hpp>


#include"Library/DataFormat/DataFormatStringArray/DataFormat.h"
#include"Library/NeuralNetwork/LayerDLLManager/LayerDLLManager.h"
#include"Layer/IOData/IODataLayer/IODataLayer.h"

using namespace Gravisbell;

/** サンプルデータの読み込み */
DataFormat::IDataFormatBase* LoadSampleData(const std::wstring& formatFilePath, const std::wstring& dataFilePath);
/** レイヤーDLL管理クラスの作成 */
Layer::NeuralNetwork::ILayerDLLManager* CreateLayerDLLManager(void);

int _tmain(int argc, _TCHAR* argv[])
{
#ifdef _DEBUG
	::_CrtSetDbgFlag(_CRTDBG_LEAK_CHECK_DF | _CRTDBG_ALLOC_MEM_DF);
#endif


	// サンプルデータの読み込み
	Gravisbell::DataFormat::IDataFormatBase* pSampleData = ::LoadSampleData(L"DataFormat.xml", L"../../SampleData/crx.csv");

	// レイヤーDLL管理クラスを作成
	Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager = ::CreateLayerDLLManager();
	if(pLayerDLLManager == NULL)
	{
		delete pSampleData;
	}

	// 入出力信号レイヤーを作成
	Layer::IOData::IIODataLayer* pInputLayer  = Layer::IOData::CreateIODataLayerCPU( pSampleData->GetDataStruct(L"input") );	// 入力信号
	Layer::IOData::IIODataLayer* pOutputLayer = Layer::IOData::CreateIODataLayerCPU( pSampleData->GetDataStruct(L"output") );	// 出力信号
	for(U32 dataNum=0; dataNum<pSampleData->GetDataCount(); dataNum++)
	{
		pInputLayer->AddData(pSampleData->GetDataByNum(dataNum, L"input"));
		pOutputLayer->AddData(pSampleData->GetDataByNum(dataNum, L"output"));
	}

	// メモリ開放
	delete pInputLayer;
	delete pOutputLayer;
	delete pLayerDLLManager;
	delete pSampleData;

	return 0;
}


/** サンプルデータの読み込み */
Gravisbell::DataFormat::IDataFormatBase* LoadSampleData(const std::wstring& formatFilePath, const std::wstring& dataFilePath)
{
	// フォーマットを読み込み
	auto pDataFormat = Gravisbell::DataFormat::StringArray::CreateDataFormatFromXML(formatFilePath.c_str());
	if(pDataFormat == NULL)
		return NULL;

	// CSVファイルを読み込んでフォーマットに追加
	{
		// ファイルオープン
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

			// ","(カンマ)区切りで分離
			std::vector<std::wstring> lpBuf;
			boost::split(lpBuf, szBuf, boost::is_any_of(L","));

			std::vector<const wchar_t*> lpBufPointer;
			for(auto& buf : lpBuf)
				lpBufPointer.push_back(buf.c_str());


			pDataFormat->AddDataByStringArray(&lpBufPointer[0]);
		}

		// ファイルクローズ
		fclose(fp);
	}
	// 正規化
	pDataFormat->Normalize();

	return pDataFormat;
}
/** レイヤーDLL管理クラスの作成 */
Layer::NeuralNetwork::ILayerDLLManager* CreateLayerDLLManager(void)
{
	// DLL管理クラスを作成
	Layer::NeuralNetwork::ILayerDLLManager* pDLLManager = Layer::NeuralNetwork::CreateLayerDLLManagerCPU();

	// DLLの読み込み.
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
