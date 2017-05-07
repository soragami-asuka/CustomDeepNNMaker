//==================================
// ニューラルネットワークのレイヤー管理用のUtiltiy
// ライブラリとして使う間は有効.
// ツール化後は消す予定
//==================================
#include"stdafx.h"

#include"NeuralNetworkLayer.h"

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
Layer::NeuralNetwork::INNLayerConnectData* CreateNeuralNetwork(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct)
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
	Layer::NeuralNetwork::INNLayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, inputDataStruct);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	// キャスト
	Layer::NeuralNetwork::INNLayerConnectData* pNeuralNetwork = dynamic_cast<Layer::NeuralNetwork::INNLayerConnectData*>(pLayer);
	if(pNeuralNetwork == NULL)
	{
		delete pLayer;
		return NULL;
	}

	return pNeuralNetwork;
}
Layer::NeuralNetwork::INNLayerData* CreateConvolutionLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, Vector3D<S32> filterSize, U32 outputChannelCount, Vector3D<S32> stride, Vector3D<S32> paddingSize)
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
	Layer::NeuralNetwork::INNLayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, inputDataStruct);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}
Layer::NeuralNetwork::INNLayerData* CreateFullyConnectLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, U32 neuronCount)
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
	Layer::NeuralNetwork::INNLayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, inputDataStruct);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}
Layer::NeuralNetwork::INNLayerData* CreateActivationLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, const std::wstring activationType)
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
	Layer::NeuralNetwork::INNLayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, inputDataStruct);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}
Layer::NeuralNetwork::INNLayerData* CreateDropoutLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, F32 rate)
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
	Layer::NeuralNetwork::INNLayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, inputDataStruct);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}
Layer::NeuralNetwork::INNLayerData* CreatePoolingLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, Vector3D<S32> filterSize)
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

	// レイヤーの作成
	Layer::NeuralNetwork::INNLayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, inputDataStruct);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}
Layer::NeuralNetwork::INNLayerData* CreateBatchNormalizationLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct)
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
	Layer::NeuralNetwork::INNLayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, inputDataStruct);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}


/** レイヤーをネットワークの末尾に追加する.GUIDは自動割り当て.入力データ構造、最終GUIDも更新する. */
Gravisbell::ErrorCode AddLayerToNetworkLast( Layer::NeuralNetwork::INNLayerConnectData& neuralNetwork, std::list<Layer::ILayerData*>& lppLayerData, Gravisbell::IODataStruct& inputDataStruct, Gravisbell::GUID& lastLayerGUID, Layer::NeuralNetwork::INNLayerData* pAddlayer)
{
	// GUID生成
	Gravisbell::GUID guid = boost::uuids::random_generator()().data;

	lppLayerData.push_back(pAddlayer);
	neuralNetwork.AddLayer(guid, pAddlayer);

	// 接続
	neuralNetwork.AddInputLayerToLayer(guid, lastLayerGUID);

	// 現在レイヤーを直前レイヤーに変更
	inputDataStruct = pAddlayer->GetOutputDataStruct();
	lastLayerGUID = guid;

	return Gravisbell::ErrorCode::ERROR_CODE_NONE;
}


}	// NeuralNetworkLayer
}	// Utility
}	// Gravisbell
