//======================================
// フィードフォワードニューラルネットワークの処理レイヤーのデータ
// 複数のレイヤーを内包し、処理する
//======================================
#include"stdafx.h"

#include"FeedforwardNeuralNetwork_FUNC.hpp"
#include"FeedforwardNeuralNetwork_LayerData_Base.h"
#include"FeedforwardNeuralNetwork_CPU.h"

using namespace Gravisbell;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class FeedforwardNeuralNetwork_LayerData_CPU : public FeedforwardNeuralNetwork_LayerData_Base
	{	
		//====================================
		// コンストラクタ/デストラクタ
		//====================================
	public:
		/** コンストラクタ */
		FeedforwardNeuralNetwork_LayerData_CPU(const ILayerDLLManager& i_layerDLLManager, const Gravisbell::GUID& guid)
			:	FeedforwardNeuralNetwork_LayerData_Base(i_layerDLLManager, guid)
		{
		}
		/** デストラクタ */
		virtual ~FeedforwardNeuralNetwork_LayerData_CPU()
		{
		}

		//===========================
		// レイヤー作成
		//===========================
		/** レイヤーを作成する.
			@param guid	新規生成するレイヤーのGUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
		{
			if(this->CheckCanUseInputDataStruct(i_lpInputDataStruct, i_inputLayerCount) == false)
				return NULL;

			FeedforwardNeuralNetwork_Base* pNeuralNetwork = new FeedforwardNeuralNetwork_CPU(guid, *this, i_lpInputDataStruct[0]);

			// ニューラルネットワークにレイヤーを追加
			ErrorCode err = AddConnectionLayersToNeuralNetwork(*pNeuralNetwork, i_lpInputDataStruct, i_inputLayerCount);
			if(err != ErrorCode::ERROR_CODE_NONE)
			{
				delete pNeuralNetwork;
				return NULL;
			}

			return pNeuralNetwork;
		}
		/** レイヤーを作成する.
			@param guid	新規生成するレイヤーのGUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		{
			if(this->CheckCanUseInputDataStruct(i_lpInputDataStruct, i_inputLayerCount) == false)
				return NULL;

			FeedforwardNeuralNetwork_Base* pNeuralNetwork = new FeedforwardNeuralNetwork_CPU(guid, *this, i_lpInputDataStruct[0], i_temporaryMemoryManager);

			// ニューラルネットワークにレイヤーを追加
			ErrorCode err = AddConnectionLayersToNeuralNetwork(*pNeuralNetwork, i_lpInputDataStruct, i_inputLayerCount);
			if(err != ErrorCode::ERROR_CODE_NONE)
			{
				delete pNeuralNetwork;
				return NULL;
			}

			return pNeuralNetwork;
		}
	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell


/** Create a layer for CPU processing.
  * @param GUID of layer to create.
  */
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataCPU(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const Gravisbell::SettingData::Standard::IData& i_data)
{
	// DLLマネージャのNULLチェック
	if(pLayerDLLManager == NULL)
		return NULL;

	// 作成
	Gravisbell::Layer::NeuralNetwork::FeedforwardNeuralNetwork_LayerData_CPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::FeedforwardNeuralNetwork_LayerData_CPU(*pLayerDLLManager, guid);
	if(pLayerData == NULL)
		return NULL;

	// 初期化
	Gravisbell::ErrorCode errCode = pLayerData->Initialize(i_data);
	if(errCode != Gravisbell::ErrorCode::ERROR_CODE_NONE)
	{
		delete pLayerData;
		return NULL;
	}

	return pLayerData;
}
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataCPUfromBuffer(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const BYTE* i_lpBuffer, S32 i_bufferSize, S32& o_useBufferSize)
{
	// DLLマネージャのNULLチェック
	if(pLayerDLLManager == NULL)
		return NULL;

	// 作成
	Gravisbell::Layer::NeuralNetwork::FeedforwardNeuralNetwork_LayerData_CPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::FeedforwardNeuralNetwork_LayerData_CPU(*pLayerDLLManager, guid);
	if(pLayerData == NULL)
		return NULL;

	// 初期化
	S32 useBufferSize = 0;
	Gravisbell::ErrorCode errCode = pLayerData->InitializeFromBuffer(i_lpBuffer, i_bufferSize, useBufferSize);
	if(errCode != Gravisbell::ErrorCode::ERROR_CODE_NONE)
	{
		delete pLayerData;
		return NULL;
	}

	// 使用したバッファ量を格納
	o_useBufferSize = useBufferSize;

	return pLayerData;
}
