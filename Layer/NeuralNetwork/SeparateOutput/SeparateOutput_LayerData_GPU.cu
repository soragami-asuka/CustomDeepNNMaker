//======================================
// 出力信号分割レイヤーのデータ
// GPU制御
//======================================
#include"stdafx.h"

#include"SeparateOutput_LayerData_GPU.cuh"
#include"SeparateOutput_FUNC.hpp"
#include"SeparateOutput_GPU.cuh"

using namespace Gravisbell;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	//===========================
	// コンストラクタ / デストラクタ
	//===========================
	/** コンストラクタ */
	SeparateOutput_LayerData_GPU::SeparateOutput_LayerData_GPU(const Gravisbell::GUID& guid)
		:	SeparateOutput_LayerData_Base(guid)
	{
	}
	/** デストラクタ */
	SeparateOutput_LayerData_GPU::~SeparateOutput_LayerData_GPU()
	{
	}


	//===========================
	// レイヤー作成
	//===========================
	/** レイヤーを作成する.
		@param guid	新規生成するレイヤーのGUID. */
	ILayerBase* SeparateOutput_LayerData_GPU::CreateLayer(const Gravisbell::GUID& guid)
	{
		return new SeparateOutput_GPU(guid, *this);
	}

} // Gravisbell;
} // Layer;
} // NeuralNetwork;


/** Create a layer for GPU processing.
  * @param GUID of layer to create.
  */
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataGPU(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const Gravisbell::SettingData::Standard::IData& i_data, const Gravisbell::IODataStruct& i_inputDataStruct)
{
	// 作成
	Gravisbell::Layer::NeuralNetwork::SeparateOutput_LayerData_GPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::SeparateOutput_LayerData_GPU(guid);
	if(pLayerData == NULL)
		return NULL;

	// 初期化
	Gravisbell::ErrorCode errCode = pLayerData->Initialize(i_data, i_inputDataStruct);
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
	Gravisbell::Layer::NeuralNetwork::SeparateOutput_LayerData_GPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::SeparateOutput_LayerData_GPU(guid);
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
