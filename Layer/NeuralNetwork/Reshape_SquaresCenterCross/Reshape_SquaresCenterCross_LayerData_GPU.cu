//======================================
// 出力信号分割レイヤーのデータ
// GPU制御
//======================================
#include"stdafx.h"

#include"Reshape_SquaresCenterCross_LayerData_GPU.cuh"
#include"Reshape_SquaresCenterCross_FUNC.hpp"
#include"Reshape_SquaresCenterCross_GPU.cuh"

using namespace Gravisbell;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	//===========================
	// コンストラクタ / デストラクタ
	//===========================
	/** コンストラクタ */
	Reshape_SquaresCenterCross_LayerData_GPU::Reshape_SquaresCenterCross_LayerData_GPU(const Gravisbell::GUID& guid)
		:	Reshape_SquaresCenterCross_LayerData_Base(guid)
	{
	}
	/** デストラクタ */
	Reshape_SquaresCenterCross_LayerData_GPU::~Reshape_SquaresCenterCross_LayerData_GPU()
	{
	}


	//===========================
	// レイヤー作成
	//===========================
	/** レイヤーを作成する.
		@param guid	新規生成するレイヤーのGUID. */
	ILayerBase* Reshape_SquaresCenterCross_LayerData_GPU::CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
	{
		if(this->CheckCanUseInputDataStruct(i_lpInputDataStruct, i_inputLayerCount) == false)
			return NULL;

		return new Reshape_SquaresCenterCross_GPU(guid, *this, i_lpInputDataStruct[0]);
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
	Gravisbell::Layer::NeuralNetwork::Reshape_SquaresCenterCross_LayerData_GPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::Reshape_SquaresCenterCross_LayerData_GPU(guid);
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
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataGPUfromBuffer(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const BYTE* i_lpBuffer, S32 i_bufferSize, S32& o_useBufferSize)
{
	// 作成
	Gravisbell::Layer::NeuralNetwork::Reshape_SquaresCenterCross_LayerData_GPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::Reshape_SquaresCenterCross_LayerData_GPU(guid);
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
