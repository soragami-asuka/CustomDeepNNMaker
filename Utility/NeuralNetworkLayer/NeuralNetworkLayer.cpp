//==================================
// ニューラルネットワークのレイヤー管理用のUtiltiy
// ライブラリとして使う間は有効.
// ツール化後は消す予定
//==================================
#include"stdafx.h"

#include"NeuralNetworkLayer.h"
#include"Layer/IO/ISingleInputLayerData.h"
#include"Layer/IO/ISingleOutputLayerData.h"

#include<boost/uuid/uuid_generators.hpp>
#include<boost/foreach.hpp>

namespace Gravisbell {
namespace Utility {
namespace NeuralNetworkLayer {


/** レイヤーDLL管理クラスの作成 */
Layer::NeuralNetwork::ILayerDLLManager* CreateLayerDLLManagerCPU(const boost::filesystem::wpath& libraryDirPath)
{
	// DLL管理クラスを作成
	Layer::NeuralNetwork::ILayerDLLManager* pDLLManager = Layer::NeuralNetwork::CreateLayerDLLManagerCPU();

	BOOST_FOREACH(const boost::filesystem::wpath& path, std::make_pair(boost::filesystem::directory_iterator(libraryDirPath),
                                                    boost::filesystem::directory_iterator()))
	{
		if(path.stem().wstring().find(L"Gravisbell.Layer.NeuralNetwork.") == 0 && path.extension().wstring()==L".dll")
		{
			pDLLManager->ReadLayerDLL(path.generic_wstring().c_str());
		}
    }
	return pDLLManager;
}
Layer::NeuralNetwork::ILayerDLLManager* CreateLayerDLLManagerGPU(const boost::filesystem::wpath& libraryDirPath)
{
	// DLL管理クラスを作成
	Layer::NeuralNetwork::ILayerDLLManager* pDLLManager = Layer::NeuralNetwork::CreateLayerDLLManagerGPU();

	BOOST_FOREACH(const boost::filesystem::wpath& path, std::make_pair(boost::filesystem::directory_iterator(libraryDirPath),
                                                    boost::filesystem::directory_iterator()))
	{
		if(path.stem().wstring().find(L"Gravisbell.Layer.NeuralNetwork.") == 0 && path.extension().wstring()==L".dll")
		{
			pDLLManager->ReadLayerDLL(path.generic_wstring().c_str());
		}
    }
	return pDLLManager;
}


/** レイヤーデータを作成 */
Layer::Connect::ILayerConnectData* CreateNeuralNetwork(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct)
{
	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(Gravisbell::GUID(0x1c38e21f, 0x6f01, 0x41b2, 0xb4, 0x0e, 0x7f, 0x67, 0x26, 0x7a, 0x36, 0x92));
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// レイヤーの作成
	Layer::ILayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, inputDataStruct);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	// キャスト
	Layer::Connect::ILayerConnectData* pNeuralNetwork = dynamic_cast<Layer::Connect::ILayerConnectData*>(pLayer);
	if(pNeuralNetwork == NULL)
	{
		delete pLayer;
		return NULL;
	}

	return pNeuralNetwork;
}
Layer::ILayerData* CreateConvolutionLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, Vector3D<S32> filterSize, U32 outputChannelCount, Vector3D<S32> stride, Vector3D<S32> paddingSize)
{
	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(Gravisbell::GUID(0xf6662e0e, 0x1ca4, 0x4d59, 0xac, 0xca, 0xca, 0xc2, 0x9a, 0x16, 0xc0, 0xaa));
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;
	// フィルタサイズ
	{
		SettingData::Standard::IItem_Vector3D_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Vector3D_Int*>(pConfig->GetItemByID(L"FilterSize"));
		pItem->SetValue(filterSize);
	}
	// 出力チャンネル数
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"Output_Channel"));
		pItem->SetValue(outputChannelCount);
	}
	// フィルタ移動量
	{
		SettingData::Standard::IItem_Vector3D_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Vector3D_Int*>(pConfig->GetItemByID(L"Stride"));
		pItem->SetValue(stride);
	}
	// パディングサイズ
	{
		SettingData::Standard::IItem_Vector3D_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Vector3D_Int*>(pConfig->GetItemByID(L"Padding"));
		pItem->SetValue(paddingSize);
	}
	// パディング種別
	{
		SettingData::Standard::IItem_Enum* pItem = dynamic_cast<SettingData::Standard::IItem_Enum*>(pConfig->GetItemByID(L"PaddingType"));
		pItem->SetValue(L"zero");
	}

	// レイヤーの作成
	Layer::ILayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, inputDataStruct);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}
Layer::ILayerData* CreateFullyConnectLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, U32 neuronCount)
{
	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(Gravisbell::GUID(0x14cc33f4, 0x8cd3, 0x4686, 0x9c, 0x48, 0xef, 0x45, 0x2b, 0xa5, 0xd2, 0x02));
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;
	// ニューロン数
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"NeuronCount"));
		pItem->SetValue(neuronCount);
	}

	// レイヤーの作成
	Layer::ILayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, inputDataStruct);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}
Layer::ILayerData* CreateActivationLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, const std::wstring activationType)
{
	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(Gravisbell::GUID(0x99904134, 0x83b7, 0x4502, 0xa0, 0xca, 0x72, 0x8a, 0x2c, 0x9d, 0x80, 0xc7));
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;
	// 活性化関数種別
	{
		SettingData::Standard::IItem_Enum* pItem = dynamic_cast<SettingData::Standard::IItem_Enum*>(pConfig->GetItemByID(L"ActivationType"));
		pItem->SetValue(activationType.c_str());
	}

	// レイヤーの作成
	Layer::ILayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, inputDataStruct);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}
Layer::ILayerData* CreateDropoutLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, F32 rate)
{
	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(Gravisbell::GUID(0x298243e4, 0x2111, 0x474f, 0xa8, 0xf4, 0x35, 0xbd, 0xc8, 0x76, 0x45, 0x88));
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;
	// 活性化関数種別
	{
		SettingData::Standard::IItem_Float* pItem = dynamic_cast<SettingData::Standard::IItem_Float*>(pConfig->GetItemByID(L"Rate"));
		pItem->SetValue(rate);
	}

	// レイヤーの作成
	Layer::ILayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, inputDataStruct);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}
Layer::ILayerData* CreatePoolingLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, Vector3D<S32> filterSize, Vector3D<S32> stride)
{
	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(Gravisbell::GUID(0xeb80e0d0, 0x9d5a, 0x4ed1, 0xa8, 0x0d, 0xa1, 0x66, 0x7d, 0xe0, 0xc8, 0x90));
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;
	// フィルタサイズ
	{
		SettingData::Standard::IItem_Vector3D_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Vector3D_Int*>(pConfig->GetItemByID(L"FilterSize"));
		pItem->SetValue(filterSize);
	}
	// ストライド
	{
		SettingData::Standard::IItem_Vector3D_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Vector3D_Int*>(pConfig->GetItemByID(L"Stride"));
		pItem->SetValue(stride);
	}

	// レイヤーの作成
	Layer::ILayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, inputDataStruct);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}
Layer::ILayerData* CreateBatchNormalizationLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct)
{
	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(Gravisbell::GUID(0xacd11a5a, 0xbfb5, 0x4951, 0x83, 0x82, 0x1d, 0xe8, 0x9d, 0xfa, 0x96, 0xa8));
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;
	
	// レイヤーの作成
	Layer::ILayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, inputDataStruct);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}
Layer::ILayerData* CreateGlobalAveragePoolingLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct)
{
	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(Gravisbell::GUID(0xf405d6d7, 0x434c, 0x4ed2, 0x82, 0xc3, 0x5d, 0x7e, 0x49, 0xf4, 0x03, 0xdb));
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;
	
	// レイヤーの作成
	Layer::ILayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, inputDataStruct);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}

Layer::ILayerData* CreateMergeInputLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct lpInputDataStruct[], U32 inputDataCount)
{
	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(Gravisbell::GUID(0x53daec93, 0xdbdb, 0x4048, 0xbd, 0x5a, 0x40, 0x1d, 0xd0, 0x05, 0xc7, 0x4e));
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;
	
	// レイヤーの作成
	Layer::ILayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, lpInputDataStruct, inputDataCount);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}
Layer::ILayerData* CreateMergeInputLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const std::vector<IODataStruct>& lpInputDataStruct)
{
	return CreateMergeInputLayer(layerDLLManager, &lpInputDataStruct[0], (U32)lpInputDataStruct.size());
}


Layer::ILayerData* CreateResidualLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct lpInputDataStruct[], U32 inputDataCount)
{
	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(Gravisbell::GUID(0x0519e7fa, 0x311d, 0x4a1d, 0xa6, 0x15, 0x95, 0x9a, 0xfd, 0xd0, 0x05, 0x26));
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// レイヤーの作成
	Layer::ILayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, lpInputDataStruct, inputDataCount);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}
Layer::ILayerData* CreateResidualLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const std::vector<IODataStruct>& lpInputDataStruct)
{
	return CreateResidualLayer(layerDLLManager, &lpInputDataStruct[0], (U32)lpInputDataStruct.size());
}



/** レイヤーをネットワークの末尾に追加する.GUIDは自動割り当て.入力データ構造、最終GUIDも更新する. */
Gravisbell::ErrorCode AddLayerToNetworkLast( Layer::Connect::ILayerConnectData& neuralNetwork, std::list<Layer::ILayerData*>& lppLayerData, Gravisbell::IODataStruct& inputDataStruct, Gravisbell::GUID& lastLayerGUID, Layer::ILayerData* pAddlayer)
{
	if(Layer::IO::ISingleOutputLayerData* pOutputLayerData = dynamic_cast<Layer::IO::ISingleOutputLayerData*>(pAddlayer))
	{
		// GUID生成
		Gravisbell::GUID guid = boost::uuids::random_generator()().data;

		lppLayerData.push_back(pAddlayer);
		neuralNetwork.AddLayer(guid, pAddlayer);

		// 接続
		neuralNetwork.AddInputLayerToLayer(guid, lastLayerGUID);

		// 現在レイヤーを直前レイヤーに変更
		inputDataStruct = pOutputLayerData->GetOutputDataStruct();
		lastLayerGUID = guid;

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}

	return Gravisbell::ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
}

Gravisbell::ErrorCode AddLayerToNetworkLast(
	Layer::Connect::ILayerConnectData& neuralNetwork,
	std::list<Layer::ILayerData*>& lppLayerData, Gravisbell::IODataStruct& inputDataStruct, Gravisbell::GUID& lastLayerGUID, Layer::ILayerData* pAddLayer,
	const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount)
{
	if(Layer::IO::ISingleOutputLayerData* pOutputLayerData = dynamic_cast<Layer::IO::ISingleOutputLayerData*>(pAddLayer))
	{
		// GUID生成
		Gravisbell::GUID guid = boost::uuids::random_generator()().data;

		lppLayerData.push_back(pAddLayer);
		neuralNetwork.AddLayer(guid, pAddLayer);

		// 接続
		for(U32 inputNum=0; inputNum<inputLayerCount; inputNum++)
			neuralNetwork.AddInputLayerToLayer(guid, lpInputLayerGUID[inputNum]);

		// 現在レイヤーを直前レイヤーに変更
		inputDataStruct = pOutputLayerData->GetOutputDataStruct();
		lastLayerGUID = guid;

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}

	return Gravisbell::ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
}
Gravisbell::ErrorCode AddLayerToNetworkLast(
	Layer::Connect::ILayerConnectData& neuralNetwork,
	std::list<Layer::ILayerData*>& lppLayerData, Gravisbell::IODataStruct& inputDataStruct, Gravisbell::GUID& lastLayerGUID, Layer::ILayerData* pAddLayer,
	const std::vector<Gravisbell::GUID>& lpInputLayerGUID)
{
	return AddLayerToNetworkLast(neuralNetwork, lppLayerData, inputDataStruct, lastLayerGUID, pAddLayer, &lpInputLayerGUID[0], (U32)lpInputLayerGUID.size());
}



/** ニューラルネットワークをバイナリファイルに保存する */
Gravisbell::ErrorCode WriteNetworkToBinaryFile(const Layer::ILayerData& neuralNetwork, const boost::filesystem::path& filePath)
{
	// バッファを用意する
	std::vector<BYTE> lpBuffer;
	S32 writeByteCount = 0;
	lpBuffer.resize(sizeof(Gravisbell::GUID) + neuralNetwork.GetUseBufferByteCount());

	// レイヤー種別を書き込む
	Gravisbell::GUID typeCode = neuralNetwork.GetLayerCode();
	memcpy(&lpBuffer[writeByteCount], &typeCode, sizeof(Gravisbell::GUID));
	writeByteCount += sizeof(Gravisbell::GUID);

	// バッファへ読み込む
	writeByteCount += neuralNetwork.WriteToBuffer(&lpBuffer[writeByteCount]);
	if(writeByteCount != lpBuffer.size())
		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;

	// バッファをファイルへ書き込む
	{
		// ファイルオープン
		FILE* fp = fopen(filePath.string().c_str(), "wb");
		if(fp == NULL)
			return ErrorCode::ERROR_CODE_COMMON_FILE_NOT_FOUND;

		// 書き込み
		fwrite(&lpBuffer[0], 1, lpBuffer.size(),fp);

		// ファイルクローズ
		fclose(fp);
	}

	return ErrorCode::ERROR_CODE_NONE;
}
/** ニューラルネットワークをバイナリファイルから読み込むする */
Gravisbell::ErrorCode ReadNetworkFromBinaryFile(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::ILayerData** ppNeuralNetwork, const boost::filesystem::path& filePath)
{
	std::vector<BYTE> lpBuffer;
	S32 readByteCount = 0;

	// ファイルの中身をバッファにコピーする
	{
		// ファイルオープン
		FILE* fp = fopen(filePath.string().c_str(), "rb");
		if(fp == NULL)
			return ErrorCode::ERROR_CODE_COMMON_FILE_NOT_FOUND;

		// ファイルサイズを調べてバッファを作成する
		fseek(fp, 0, SEEK_END);
		U32 fileSize = ftell(fp);
		lpBuffer.resize(fileSize);

		// 読込
		fseek(fp, 0, SEEK_SET);
		fread(&lpBuffer[0], 1, fileSize, fp);

		// ファイルクローズ
		fclose(fp);
	}

	// 種別コードを読み込む
	Gravisbell::GUID typeCode;
	memcpy(&typeCode, &lpBuffer[readByteCount], sizeof(Gravisbell::GUID));
	readByteCount += sizeof(Gravisbell::GUID);

	// DLLを取得
	auto pLayerDLL = layerDLLManager.GetLayerDLLByGUID(typeCode);
	if(pLayerDLL == NULL)
		return ErrorCode::ERROR_CODE_DLL_NOTFOUND;

	// ネットワークを作成
	S32 useBufferCount = 0;
	*ppNeuralNetwork = pLayerDLL->CreateLayerDataFromBuffer(&lpBuffer[readByteCount], (S32)lpBuffer.size()-readByteCount, useBufferCount);
	readByteCount += useBufferCount;

	return ErrorCode::ERROR_CODE_NONE;
}


}	// NeuralNetworkLayer
}	// Utility
}	// Gravisbell
