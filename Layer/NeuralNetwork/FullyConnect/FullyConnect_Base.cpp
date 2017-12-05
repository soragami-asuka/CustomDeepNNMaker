//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化を処理する
//======================================
#include"stdafx.h"

#include"FullyConnect_FUNC.hpp"

#include"FullyConnect_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
FullyConnect_Base::FullyConnect_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, ::CreateRuntimeParameter(), i_inputDataStruct, i_outputDataStruct)
{
}

/** デストラクタ */
FullyConnect_Base::~FullyConnect_Base()
{
}


//===========================
// 固有関数
//===========================
/** ニューロン数を取得する */
U32 FullyConnect_Base::GetNeuronCount()const
{
	const FullyConnect_LayerData_Base* pLayerData = dynamic_cast<const FullyConnect_LayerData_Base*>(&this->GetLayerData());
	if(pLayerData == NULL)
		return NULL;

	return pLayerData->GetNeuronCount();
}
