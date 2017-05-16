//======================================
// 出力信号分割レイヤー
//======================================
#include"stdafx.h"

#include"SeparateOutput_FUNC.hpp"

#include"SeparateOutput_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
SeparateOutput_Base::SeparateOutput_Base(Gravisbell::GUID guid)
	:	guid				(guid)
	,	pLearnData			(NULL)
{
}

/** デストラクタ */
SeparateOutput_Base::~SeparateOutput_Base()
{
	if(pLearnData != NULL)
		delete pLearnData;
}


//===========================
// レイヤー共通
//===========================

/** レイヤー種別の取得.
	ELayerKind の組み合わせ. */
U32 SeparateOutput_Base::GetLayerKindBase(void)const
{
	return Gravisbell::Layer::ELayerKind::LAYER_KIND_NEURALNETWORK | Gravisbell::Layer::ELayerKind::LAYER_KIND_SINGLE_INPUT | Gravisbell::Layer::ELayerKind::LAYER_KIND_MULT_OUTPUT;
}

/** レイヤー固有のGUIDを取得する */
Gravisbell::GUID SeparateOutput_Base::GetGUID(void)const
{
	return this->guid;
}

/** レイヤー識別コードを取得する.
	@param o_layerCode	格納先バッファ
	@return 成功した場合0 */
Gravisbell::GUID SeparateOutput_Base::GetLayerCode(void)const
{
	return this->GetLayerData().GetLayerCode();
}

/** バッチサイズを取得する.
	@return 同時に演算を行うバッチのサイズ */
unsigned int SeparateOutput_Base::GetBatchSize()const
{
	return this->batchSize;
}

/** レイヤーの設定情報を取得する */
const SettingData::Standard::IData* SeparateOutput_Base::GetLayerStructure()const
{
	return this->GetLayerData().GetLayerStructure();
}


//===========================
// 入力レイヤー関連
//===========================
/** 入力データ構造を取得する.
	@return	入力データ構造 */
IODataStruct SeparateOutput_Base::GetInputDataStruct()const
{
	return this->GetLayerData().GetInputDataStruct();
}

/** 入力バッファ数を取得する. */
unsigned int SeparateOutput_Base::GetInputBufferCount()const
{
	return this->GetLayerData().GetInputBufferCount();
}


//===========================
// 出力レイヤー関連
//===========================
/** 出力データの出力先レイヤー数. */
U32 SeparateOutput_Base::GetOutputToLayerCount()const
{
	return this->GetLayerData().GetOutputToLayerCount();
}
/** 出力データ構造を取得する */
IODataStruct SeparateOutput_Base::GetOutputDataStruct()const
{
	return this->GetLayerData().GetOutputDataStruct();
}

/** 出力バッファ数を取得する */
unsigned int SeparateOutput_Base::GetOutputBufferCount()const
{
	return this->GetLayerData().GetOutputBufferCount();
}


//===========================
// 固有関数
//===========================
