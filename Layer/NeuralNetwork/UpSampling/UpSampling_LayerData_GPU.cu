//======================================
// 畳み込みニューラルネットワークのレイヤーデータ
// GPU制御
//======================================
#include"stdafx.h"

#include"UpSampling_LayerData_GPU.cuh"
#include"UpSampling_FUNC.hpp"
#include"UpSampling_GPU.cuh"

#include"../_LayerBase/CLayerBase_GPU.cuh"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	//===========================
	// コンストラクタ / デストラクタ
	//===========================
	/** コンストラクタ */
	UpSampling_LayerData_GPU::UpSampling_LayerData_GPU(const Gravisbell::GUID& guid)
		:	UpSampling_LayerData_Base(guid)
	{
	}
	/** デストラクタ */
	UpSampling_LayerData_GPU::~UpSampling_LayerData_GPU()
	{
	}


	//===========================
	// レイヤー作成
	//===========================
	/** レイヤーを作成する.
		@param guid	新規生成するレイヤーのGUID. */
	ILayerBase* UpSampling_LayerData_GPU::CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
	{
		if(this->CheckCanUseInputDataStruct(i_lpInputDataStruct, i_inputLayerCount) == false)
			return NULL;

		return new CNNSingle2SingleLayerBase_GPU<UpSampling_GPU,UpSampling_LayerData_GPU>(guid, *this, i_lpInputDataStruct[0], i_temporaryMemoryManager);
	}

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

using namespace Gravisbell;

/** Create a layer for GPU processing.
  * @param GUID of layer to create.
  */
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataGPU(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const Gravisbell::SettingData::Standard::IData& i_data)
{
	// 作成
	Gravisbell::Layer::NeuralNetwork::UpSampling_LayerData_GPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::UpSampling_LayerData_GPU(guid);
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
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataGPUfromBuffer(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const BYTE* i_lpBuffer, Gravisbell::S64 i_bufferSize, Gravisbell::S64& o_useBufferSize)
{
	// 作成
	Gravisbell::Layer::NeuralNetwork::UpSampling_LayerData_GPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::UpSampling_LayerData_GPU(guid);
	if(pLayerData == NULL)
		return NULL;

	// 初期化
	S64 useBufferSize = 0;
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
