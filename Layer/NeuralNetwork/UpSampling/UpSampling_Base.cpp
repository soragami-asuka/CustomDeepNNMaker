//======================================
// 畳み込みニューラルネットワークの結合レイヤー
//======================================
#include"stdafx.h"

#include"UpSampling_FUNC.hpp"

#include"UpSampling_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
UpSampling_Base::UpSampling_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	guid				(guid)
	,	inputDataStruct		(i_inputDataStruct)
	,	outputDataStruct	(i_outputDataStruct)
	,	pLearnData			(NULL)
{
}

/** デストラクタ */
UpSampling_Base::~UpSampling_Base()
{
	if(pLearnData != NULL)
		delete pLearnData;
}


//===========================
// レイヤー共通
//===========================

/** レイヤー種別の取得.
	ELayerKind の組み合わせ. */
U32 UpSampling_Base::GetLayerKindBase(void)const
{
	return Gravisbell::Layer::ELayerKind::LAYER_KIND_NEURALNETWORK | Gravisbell::Layer::ELayerKind::LAYER_KIND_SINGLE_INPUT | Gravisbell::Layer::ELayerKind::LAYER_KIND_SINGLE_OUTPUT;
}

/** レイヤー固有のGUIDを取得する */
Gravisbell::GUID UpSampling_Base::GetGUID(void)const
{
	return this->guid;
}

/** レイヤー識別コードを取得する.
	@param o_layerCode	格納先バッファ
	@return 成功した場合0 */
Gravisbell::GUID UpSampling_Base::GetLayerCode(void)const
{
	return this->GetLayerData().GetLayerCode();
}

/** バッチサイズを取得する.
	@return 同時に演算を行うバッチのサイズ */
unsigned int UpSampling_Base::GetBatchSize()const
{
	return this->batchSize;
}

/** レイヤーの設定情報を取得する */
const SettingData::Standard::IData* UpSampling_Base::GetLayerStructure()const
{
	return this->GetLayerData().GetLayerStructure();
}


//===========================
// 入力レイヤー関連
//===========================
/** 入力データ構造を取得する.
	@return	入力データ構造 */
IODataStruct UpSampling_Base::GetInputDataStruct()const
{
	return this->inputDataStruct;
}

/** 入力バッファ数を取得する. */
unsigned int UpSampling_Base::GetInputBufferCount()const
{
	return this->GetInputDataStruct().GetDataCount();
}


//===========================
// 出力レイヤー関連
//===========================
/** 出力データ構造を取得する */
IODataStruct UpSampling_Base::GetOutputDataStruct()const
{
	return this->outputDataStruct;
}

/** 出力バッファ数を取得する */
unsigned int UpSampling_Base::GetOutputBufferCount()const
{
	return this->GetOutputDataStruct().GetDataCount();
}


//===========================
// 固有関数
//===========================

