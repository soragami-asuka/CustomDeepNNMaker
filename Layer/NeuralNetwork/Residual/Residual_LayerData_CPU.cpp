//======================================
// プーリングレイヤーのデータ
// CPU制御
//======================================
#include"stdafx.h"

#include"Residual_LayerData_CPU.h"
#include"Residual_FUNC.hpp"
#include"Residual_CPU.h"

using namespace Gravisbell;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	//===========================
	// コンストラクタ / デストラクタ
	//===========================
	/** コンストラクタ */
	Residual_LayerData_CPU::Residual_LayerData_CPU(const Gravisbell::GUID& guid)
		:	Residual_LayerData_Base(guid)
	{
	}
	/** デストラクタ */
	Residual_LayerData_CPU::~Residual_LayerData_CPU()
	{
	}


	//===========================
	// レイヤー作成
	//===========================
	/** レイヤーを作成する.
		@param guid	新規生成するレイヤーのGUID. */
	ILayerBase* Residual_LayerData_CPU::CreateLayer(const Gravisbell::GUID& guid)
	{
		return new Residual_CPU(guid, *this);
	}

} // Gravisbell;
} // Layer;
} // NeuralNetwork;


/** Create a layer for CPU processing.
  * @param GUID of layer to create.
  */
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataCPU_MultInput(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const Gravisbell::SettingData::Standard::IData& i_data, const Gravisbell::IODataStruct i_inputDataStruct[], U32 i_inputDataCount)
{
	// 作成
	Gravisbell::Layer::NeuralNetwork::Residual_LayerData_CPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::Residual_LayerData_CPU(guid);
	if(pLayerData == NULL)
		return NULL;

	// 初期化
	Gravisbell::ErrorCode errCode = pLayerData->Initialize(i_data, i_inputDataStruct, i_inputDataCount);
	if(errCode != Gravisbell::ErrorCode::ERROR_CODE_NONE)
	{
		delete pLayerData;
		return NULL;
	}

	return pLayerData;
}
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataCPUfromBuffer(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const BYTE* i_lpBuffer, S32 i_bufferSize, S32& o_useBufferSize)
{
	// 作成
	Gravisbell::Layer::NeuralNetwork::Residual_LayerData_CPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::Residual_LayerData_CPU(guid);
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
