//======================================
// プーリングレイヤーのデータ
// GPU制御
//======================================
#include"stdafx.h"

#include"MergeAdd_LayerData_GPU.cuh"
#include"MergeAdd_FUNC.hpp"
#include"MergeAdd_GPU.cuh"

#include"../_LayerBase/CLayerBase_GPU.cuh"

using namespace Gravisbell;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	//===========================
	// コンストラクタ / デストラクタ
	//===========================
	/** コンストラクタ */
	MergeAdd_LayerData_GPU::MergeAdd_LayerData_GPU(const Gravisbell::GUID& guid)
		:	MergeAdd_LayerData_Base(guid)
	{
	}
	/** デストラクタ */
	MergeAdd_LayerData_GPU::~MergeAdd_LayerData_GPU()
	{
	}


	//===========================
	// レイヤー作成
	//===========================
	/** レイヤーを作成する.
		@param guid	新規生成するレイヤーのGUID. */
	ILayerBase* MergeAdd_LayerData_GPU::CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
	{
		if(this->CheckCanUseInputDataStruct(i_lpInputDataStruct, i_inputLayerCount) == false)
			return NULL;

		std::vector<IODataStruct> lpInputDataStruct;
		for(U32 i=0; i<i_inputLayerCount; i++)
		{
			lpInputDataStruct.push_back(i_lpInputDataStruct[i]);
		}

		return new CNNMult2SingleLayerBase_GPU<MergeAdd_GPU,MergeAdd_LayerData_GPU>(guid, *this, lpInputDataStruct, i_temporaryMemoryManager);
	}

} // Gravisbell;
} // Layer;
} // NeuralNetwork;


/** Create a layer for GPU processing.
  * @param GUID of layer to create.
  */
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataGPU(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const Gravisbell::SettingData::Standard::IData& i_data)
{
	// 作成
	Gravisbell::Layer::NeuralNetwork::MergeAdd_LayerData_GPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::MergeAdd_LayerData_GPU(guid);
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
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataGPUfromBuffer(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const BYTE* i_lpBuffer, S64 i_bufferSize, S64& o_useBufferSize)
{
	// 作成
	Gravisbell::Layer::NeuralNetwork::MergeAdd_LayerData_GPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::MergeAdd_LayerData_GPU(guid);
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
