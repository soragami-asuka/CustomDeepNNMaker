//======================================
// プーリングレイヤーのデータ
// CPU制御
//======================================
#include"stdafx.h"

#include"MergeAverage_LayerData_CPU.h"
#include"MergeAverage_FUNC.hpp"
#include"MergeAverage_CPU.h"

#include"../_LayerBase/CLayerBase_CPU.h"

using namespace Gravisbell;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	//===========================
	// コンストラクタ / デストラクタ
	//===========================
	/** コンストラクタ */
	MergeAverage_LayerData_CPU::MergeAverage_LayerData_CPU(const Gravisbell::GUID& guid)
		:	MergeAverage_LayerData_Base(guid)
	{
	}
	/** デストラクタ */
	MergeAverage_LayerData_CPU::~MergeAverage_LayerData_CPU()
	{
	}


	//===========================
	// レイヤー作成
	//===========================
	/** レイヤーを作成する.
		@param guid	新規生成するレイヤーのGUID. */
	ILayerBase* MergeAverage_LayerData_CPU::CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
	{
		if(this->CheckCanUseInputDataStruct(i_lpInputDataStruct, i_inputLayerCount) == false)
			return NULL;

		std::vector<IODataStruct> lpInputDataStruct;
		for(U32 i=0; i<i_inputLayerCount; i++)
		{
			lpInputDataStruct.push_back(i_lpInputDataStruct[i]);
		}

		return new CNNMult2SingleLayerBase_CPU<MergeAverage_CPU,MergeAverage_LayerData_CPU>(guid, *this, lpInputDataStruct, i_temporaryMemoryManager);
	}

} // Gravisbell;
} // Layer;
} // NeuralNetwork;


/** Create a layer for CPU processing.
  * @param GUID of layer to create.
  */
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataCPU(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const Gravisbell::SettingData::Standard::IData& i_data)
{
	// 作成
	Gravisbell::Layer::NeuralNetwork::MergeAverage_LayerData_CPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::MergeAverage_LayerData_CPU(guid);
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
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataCPUfromBuffer(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const BYTE* i_lpBuffer, S64 i_bufferSize, S64& o_useBufferSize)
{
	// 作成
	Gravisbell::Layer::NeuralNetwork::MergeAverage_LayerData_CPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::MergeAverage_LayerData_CPU(guid);
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
