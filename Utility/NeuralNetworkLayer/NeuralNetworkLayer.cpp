//==================================
// ニューラルネットワークのレイヤー管理用のUtiltiy
// ライブラリとして使う間は有効.
// ツール化後は消す予定
//==================================
#include"stdafx.h"

#include"Utility/NeuralNetworkLayer.h"
#include"Library/NeuralNetwork/LayerDLLManager.h"

#include<boost/uuid/uuid_generators.hpp>
#include<boost/foreach.hpp>


namespace Gravisbell {
namespace Utility {
namespace NeuralNetworkLayer {


/** レイヤーDLL管理クラスの作成 */
Layer::NeuralNetwork::ILayerDLLManager* CreateLayerDLLManagerCPU(const wchar_t i_libraryDirPath[])
{
	// DLL管理クラスを作成
	Layer::NeuralNetwork::ILayerDLLManager* pDLLManager = Layer::NeuralNetwork::CreateLayerDLLManagerCPU();

	BOOST_FOREACH(const boost::filesystem::wpath& path, std::make_pair(boost::filesystem::directory_iterator(i_libraryDirPath),
                                                    boost::filesystem::directory_iterator()))
	{
		if(path.stem().wstring().find(L"Gravisbell.Layer.NeuralNetwork.") == 0 && path.extension().wstring()==L".dll")
		{
			pDLLManager->ReadLayerDLL(path.generic_wstring().c_str());
		}
    }
	return pDLLManager;
}
Layer::NeuralNetwork::ILayerDLLManager* CreateLayerDLLManagerGPU(const wchar_t i_libraryDirPath[])
{
	// DLL管理クラスを作成
	Layer::NeuralNetwork::ILayerDLLManager* pDLLManager = Layer::NeuralNetwork::CreateLayerDLLManagerGPU();

	BOOST_FOREACH(const boost::filesystem::wpath& path, std::make_pair(boost::filesystem::directory_iterator(i_libraryDirPath),
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
Layer::Connect::ILayerConnectData* CreateNeuralNetwork(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager)
{
	const Gravisbell::GUID TYPE_CODE(0x1c38e21f, 0x6f01, 0x41b2, 0xb4, 0x0e, 0x7f, 0x67, 0x26, 0x7a, 0x36, 0x92);

	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// レイヤーの作成
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
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
Layer::ILayerData* CreateConvolutionLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	U32 inputChannelCount, Vector3D<S32> filterSize, U32 outputChannelCount, Vector3D<S32> stride, Vector3D<S32> paddingSize,
	const wchar_t i_szInitializerID[])
{
	const Gravisbell::GUID TYPE_CODE(0xf6662e0e, 0x1ca4, 0x4d59, 0xac, 0xca, 0xca, 0xc2, 0x9a, 0x16, 0xc0, 0xaa);

	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;
	// 入力チャンネル数
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"Input_Channel"));
		pItem->SetValue(inputChannelCount);
	}
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
	// 初期化方法
	{
		SettingData::Standard::IItem_String* pItem = dynamic_cast<SettingData::Standard::IItem_String*>(pConfig->GetItemByID(L"Initializer"));
		pItem->SetValue(i_szInitializerID);
	}

	// レイヤーの作成
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}
Layer::ILayerData* CreateFullyConnectLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	U32 inputBufferCount, U32 neuronCount,
	const wchar_t i_szInitializerID[])
{
	const Gravisbell::GUID TYPE_CODE(0x14cc33f4, 0x8cd3, 0x4686, 0x9c, 0x48, 0xef, 0x45, 0x2b, 0xa5, 0xd2, 0x02);


	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;
	// 入力バッファ数
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"InputBufferCount"));
		pItem->SetValue(inputBufferCount);
	}
	// ニューロン数
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"NeuronCount"));
		pItem->SetValue(neuronCount);
	}
	// 初期化方法
	{
		SettingData::Standard::IItem_String* pItem = dynamic_cast<SettingData::Standard::IItem_String*>(pConfig->GetItemByID(L"Initializer"));
		pItem->SetValue(i_szInitializerID);
	}

	// レイヤーの作成
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}
Layer::ILayerData* CreateActivationLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	const wchar_t activationType[])
{
	const Gravisbell::GUID TYPE_CODE(0x99904134, 0x83b7, 0x4502, 0xa0, 0xca, 0x72, 0x8a, 0x2c, 0x9d, 0x80, 0xc7);


	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;
	// 活性化関数種別
	{
		SettingData::Standard::IItem_Enum* pItem = dynamic_cast<SettingData::Standard::IItem_Enum*>(pConfig->GetItemByID(L"ActivationType"));
		pItem->SetValue(activationType);
	}

	// レイヤーの作成
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}
Layer::ILayerData* CreateDropoutLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	F32 rate)
{
	const Gravisbell::GUID TYPE_CODE(0x298243e4, 0x2111, 0x474f, 0xa8, 0xf4, 0x35, 0xbd, 0xc8, 0x76, 0x45, 0x88);


	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
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
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}
/** ガウスノイズレイヤー.
	@param	layerDLLManager		レイヤーDLL管理クラス.
	@param	inputDataStruct		入力データ構造.
	@param	average				発生する乱数の平均値
	@param	variance			発生する乱数の分散 */
GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
Layer::ILayerData* CreateGaussianNoiseLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, F32 average, F32 variance)
{
	const Gravisbell::GUID TYPE_CODE(0xac27c912, 0xa11d, 0x4519, 0x81, 0xa0, 0x17, 0xc0, 0x78, 0xe4, 0x43, 0x1f);


	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// 平均
	SettingData::Standard::IItem_Float* pItem_Average = dynamic_cast<SettingData::Standard::IItem_Float*>(pConfig->GetItemByID(L"Average"));
	if(pItem_Average)
	{
		pItem_Average->SetValue(average);
	}

	// 分散
	SettingData::Standard::IItem_Float* pItem_Variance = dynamic_cast<SettingData::Standard::IItem_Float*>(pConfig->GetItemByID(L"Variance"));
	if(pItem_Variance)
	{
		pItem_Variance->SetValue(variance);
	}

	// レイヤーの作成
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}

/** プーリングレイヤー.
	@param	layerDLLManager		レイヤーDLL管理クラス.
	@param	inputDataStruct		入力データ構造.
	@param	filterSize			プーリング幅.
	@param	stride				フィルタ移動量. */
Layer::ILayerData* CreatePoolingLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	Vector3D<S32> filterSize, Vector3D<S32> stride)
{
	const Gravisbell::GUID TYPE_CODE(0xeb80e0d0, 0x9d5a, 0x4ed1, 0xa8, 0x0d, 0xa1, 0x66, 0x7d, 0xe0, 0xc8, 0x90);


	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
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
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}

/** バッチ正規化レイヤー */
Layer::ILayerData* CreateBatchNormalizationLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	U32 inputChannelCount)
{
	const Gravisbell::GUID TYPE_CODE(0xacd11a5a, 0xbfb5, 0x4951, 0x83, 0x82, 0x1d, 0xe8, 0x9d, 0xfa, 0x96, 0xa8);


	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// 入力チャンネル数
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"InputChannelCount"));
		pItem->SetValue(inputChannelCount);
	}

	// レイヤーの作成
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}

/** バッチ正規化レイヤー(チャンネル区別なし)
	@param	layerDLLManager		レイヤーDLL管理クラス. */
Layer::ILayerData* CreateBatchNormalizationAllLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager)
{
	const Gravisbell::GUID TYPE_CODE(0x8aecb925, 0x8dcf, 0x4876, 0xba, 0x6a, 0x6a, 0xdb, 0xe2, 0x80, 0xd2, 0x85);


	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// レイヤーの作成
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}

/** スケール正規化レイヤー */
Layer::ILayerData* CreateNormalizationScaleLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager)
{
	const Gravisbell::GUID TYPE_CODE(0xd8c0de15, 0x5445, 0x482d, 0xbb, 0xc9, 0x00, 0x26, 0xbf, 0xa9, 0x6a, 0xdd);

	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// レイヤーの作成
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}

/** 広域平均プーリングレイヤー
	@param	layerDLLManager		レイヤーDLL管理クラス.
	@param	inputDataStruct		入力データ構造. */
Layer::ILayerData* CreateGlobalAveragePoolingLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager)
{
	const Gravisbell::GUID TYPE_CODE(0xf405d6d7, 0x434c, 0x4ed2, 0x82, 0xc3, 0x5d, 0x7e, 0x49, 0xf4, 0x03, 0xdb);


	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;
	
	// レイヤーの作成
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}

/** GANにおけるDiscriminatorの出力レイヤー
	@param	layerDLLManager		レイヤーDLL管理クラス.
	@param	inputDataStruct		入力データ構造. */
Layer::ILayerData* CreateActivationDiscriminatorLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager)
{
	const Gravisbell::GUID TYPE_CODE(0x6694e58a, 0x954c, 0x4092, 0x86, 0xc9, 0x65, 0x3d, 0x2e, 0x12, 0x4e, 0x83);


	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;
	
	// レイヤーの作成
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}

Layer::ILayerData* CreateUpSamplingLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	Vector3D<S32> upScale, bool paddingUseValue)
{
	const Gravisbell::GUID TYPE_CODE(0x14eee4a7, 0x1b26, 0x4651, 0x8e, 0xbf, 0xb1, 0x15, 0x6d, 0x62, 0xce, 0x1b);


	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// 拡大率
	{
		SettingData::Standard::IItem_Vector3D_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Vector3D_Int*>(pConfig->GetItemByID(L"UpScale"));
		pItem->SetValue(upScale);
	}
	// パディング種別
	{
		SettingData::Standard::IItem_Enum* pItem = dynamic_cast<SettingData::Standard::IItem_Enum*>(pConfig->GetItemByID(L"PaddingType"));
		if(paddingUseValue)
		{
			pItem->SetValue(L"value");
		}
		else
		{
			pItem->SetValue(L"zero");
		}
	}

	// レイヤーの作成
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}

Layer::ILayerData* CreateMergeInputLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager)
{
	const Gravisbell::GUID TYPE_CODE(0x53daec93, 0xdbdb, 0x4048, 0xbd, 0x5a, 0x40, 0x1d, 0xd0, 0x05, 0xc7, 0x4e);


	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;
	
	// レイヤーの作成
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}

/** 入力結合レイヤー(加算). 入力されたレイヤーの値を合算する. 入力データ構造はX,Y,Zで同じサイズである必要がある.
	@param	layerDLLManager		レイヤーDLL管理クラス
	@param	i_mergeType			ch数をマージさせる方法. */
Layer::ILayerData* CreateMergeAddLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	LayerMergeType i_mergeType)
{
	const Gravisbell::GUID TYPE_CODE(0x754f6bbf, 0x7931, 0x473e, 0xae, 0x82, 0x29, 0xe9, 0x99, 0xa3, 0x4b, 0x22);


	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// パディング種別
	{
		SettingData::Standard::IItem_Enum* pItem = dynamic_cast<SettingData::Standard::IItem_Enum*>(pConfig->GetItemByID(L"MergeType"));
		switch(i_mergeType)
		{
		case LayerMergeType::LYAERMERGETYPE_MAX:
			pItem->SetValue(L"max");
			break;
		case LayerMergeType::LYAERMERGETYPE_MIN:
			pItem->SetValue(L"min");
			break;
		case LayerMergeType::LYAERMERGETYPE_LAYER0:
			pItem->SetValue(L"layer0");
			break;
		}
	}

	// レイヤーの作成
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}


/** 入力結合レイヤー(平均). 入力されたレイヤーの値の平均をとる. 入力データ構造はX,Y,Zで同じサイズである必要がある.
	@param	layerDLLManager		レイヤーDLL管理クラス
	@param	i_mergeType			ch数をマージさせる方法. */
GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
Layer::ILayerData* CreateMergeAverageLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	LayerMergeType i_mergeType)
{
	const Gravisbell::GUID TYPE_CODE(0x4e993b4b, 0x9f7a, 0x4cef, 0xa4, 0xc4, 0x37, 0xb9, 0x16, 0xbf, 0xd9, 0xb2);


	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// パディング種別
	{
		SettingData::Standard::IItem_Enum* pItem = dynamic_cast<SettingData::Standard::IItem_Enum*>(pConfig->GetItemByID(L"MergeType"));
		switch(i_mergeType)
		{
		case LayerMergeType::LYAERMERGETYPE_MAX:
			pItem->SetValue(L"max");
			break;
		case LayerMergeType::LYAERMERGETYPE_MIN:
			pItem->SetValue(L"min");
			break;
		case LayerMergeType::LYAERMERGETYPE_LAYER0:
			pItem->SetValue(L"layer0");
			break;
		}
	}

	// レイヤーの作成
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}



/** 入力結合レイヤー(最大値). 入力されたレイヤーの最大値を算出する. 入力データ構造はX,Y,Zで同じサイズである必要がある.
	@param	layerDLLManager		レイヤーDLL管理クラス
	@param	i_mergeType			ch数をマージさせる方法. */
GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
Layer::ILayerData* CreateMergeMaxLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	LayerMergeType i_mergeType)
{
	const Gravisbell::GUID TYPE_CODE(0x3f015946, 0x7e88, 0x4db0, 0x91, 0xbd, 0xf4, 0x01, 0x3f, 0x21, 0x90, 0xd4);


	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// パディング種別
	{
		SettingData::Standard::IItem_Enum* pItem = dynamic_cast<SettingData::Standard::IItem_Enum*>(pConfig->GetItemByID(L"MergeType"));
		switch(i_mergeType)
		{
		case LayerMergeType::LYAERMERGETYPE_MAX:
			pItem->SetValue(L"max");
			break;
		case LayerMergeType::LYAERMERGETYPE_MIN:
			pItem->SetValue(L"min");
			break;
		case LayerMergeType::LYAERMERGETYPE_LAYER0:
			pItem->SetValue(L"layer0");
			break;
		}
	}

	// レイヤーの作成
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}


/** チャンネル抽出レイヤー. 入力されたレイヤーの特定チャンネルを抽出する. 入力/出力データ構造でX,Y,Zは同じサイズ.
	@param	startChannelNo	開始チャンネル番号.
	@param	channelCount	抽出チャンネル数. */
Layer::ILayerData* CreateChooseChannelLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	U32 startChannelNo, U32 channelCount)
{
	const Gravisbell::GUID TYPE_CODE(0x244824b3, 0xbcfc, 0x4655, 0xa9, 0x91, 0x0f, 0x61, 0x36, 0xd3, 0x7a, 0x34);

	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// 開始チャンネル番号
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"startChannelNo"));
		pItem->SetValue(startChannelNo);
	}
	// 出力チャンネル数
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"channelCount"));
		pItem->SetValue(channelCount);
	}

	// レイヤーの作成
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}


/** 出力データ構造変換レイヤー.
	@param	ch	CH数.
	@param	x	X軸.
	@param	y	Y軸.
	@param	z	Z軸. */
Layer::ILayerData* CreateReshapeLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	U32 ch, U32 x, U32 y, U32 z)
{
	const Gravisbell::GUID TYPE_CODE(0xe78e7f59, 0xd4b3, 0x45a1, 0xae, 0xeb, 0x9f, 0x2a, 0x51, 0x55, 0x47, 0x3f);

	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// CH数
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"ch"));
		pItem->SetValue(ch);
	}
	// X軸
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"x"));
		pItem->SetValue(x);
	}
	// Y軸
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"y"));
		pItem->SetValue(y);
	}
	// Z軸
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"z"));
		pItem->SetValue(z);
	}

	// レイヤーの作成
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}
/** 出力データ構造変換レイヤー.
	@param	outputDataStruct 出力データ構造 */
Layer::ILayerData* CreateReshapeLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	const IODataStruct& outputDataStruct)
{
	return CreateReshapeLayer(layerDLLManager, layerDataManager, outputDataStruct.ch, outputDataStruct.x, outputDataStruct.y, outputDataStruct.z);
}


/** X=0を中心にミラー化する*/
Layer::ILayerData* CreateReshapeMirrorXLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager)
{
	const Gravisbell::GUID TYPE_CODE(0xdfca3f81, 0xc2f1, 0x4ac6, 0xb6, 0x18, 0x81, 0x66, 0x51, 0xad, 0xdb, 0x63);

	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// レイヤーの作成
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}

/** X=0を中心に平方化する. */
Layer::ILayerData* CreateReshapeSquareCenterCrossLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager)
{
	const Gravisbell::GUID TYPE_CODE(0x5c2729d1, 0x33eb, 0x45ef, 0xab, 0xa5, 0x0c, 0x36, 0xac, 0x22, 0xd0, 0xbc);

	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// レイヤーの作成
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}

/** X=0を中心に平方化する. */
Layer::ILayerData* CreateReshapeSquareZeroSideLeftTopLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	Gravisbell::U32 x, Gravisbell::U32 y)
{
	const Gravisbell::GUID TYPE_CODE(0xf6d9c5da, 0xd583, 0x455b, 0x92, 0x54, 0x5a, 0xef, 0x3c, 0xa9, 0x02, 0x1b);

	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// X軸
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"x"));
		pItem->SetValue(x);
	}
	// Y軸
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"y"));
		pItem->SetValue(y);
	}

	// レイヤーの作成
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}



Layer::ILayerData* CreateResidualLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager)
{
	const Gravisbell::GUID TYPE_CODE(0x0519e7fa, 0x311d, 0x4a1d, 0xa6, 0x15, 0x95, 0x9a, 0xfd, 0xd0, 0x05, 0x26);


	// DLL取得
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// 設定の作成
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// レイヤーの作成
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// 設定情報を削除
	delete pConfig;

	return pLayer;
}



/** レイヤーをネットワークの末尾に追加する.GUIDは自動割り当て.入力データ構造、最終GUIDも更新する. */
Gravisbell::ErrorCode AddLayerToNetworkLast(
	Layer::Connect::ILayerConnectData& neuralNetwork,
	Gravisbell::GUID& lastLayerGUID, Layer::ILayerData* pAddlayer, bool onLayerFix)
{
	if(pAddlayer)
	{
		// GUID生成
		Gravisbell::GUID guid = boost::uuids::random_generator()().data;

		neuralNetwork.AddLayer(guid, pAddlayer, onLayerFix);

		// 接続
		neuralNetwork.AddInputLayerToLayer(guid, lastLayerGUID);

		// 現在レイヤーを直前レイヤーに変更
		lastLayerGUID = guid;

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}

	return Gravisbell::ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
}

Gravisbell::ErrorCode AddLayerToNetworkLast(
	Layer::Connect::ILayerConnectData& neuralNetwork,
	Gravisbell::GUID& lastLayerGUID, Layer::ILayerData* pAddLayer, bool onLayerFix,
	const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount)
{
	if(pAddLayer)
	{
		// GUID生成
		Gravisbell::GUID guid = boost::uuids::random_generator()().data;

		neuralNetwork.AddLayer(guid, pAddLayer, onLayerFix);

		// 接続
		for(U32 inputNum=0; inputNum<inputLayerCount; inputNum++)
			neuralNetwork.AddInputLayerToLayer(guid, lpInputLayerGUID[inputNum]);

		// 現在レイヤーを直前レイヤーに変更
		lastLayerGUID = guid;

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}

	return Gravisbell::ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
}


/** ニューラルネットワークをバイナリファイルに保存する */
Gravisbell::ErrorCode WriteNetworkToBinaryFile(const Layer::ILayerData& neuralNetwork, const wchar_t i_filePath[])
{
	boost::filesystem::path filePath = i_filePath;

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
Gravisbell::ErrorCode ReadNetworkFromBinaryFile(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::ILayerData** ppNeuralNetwork, const wchar_t i_filePath[])
{
	boost::filesystem::path filePath = i_filePath;

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
