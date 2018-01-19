//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化を処理する
//======================================
#include"stdafx.h"

#include"SOM_FUNC.hpp"

#include"SOM_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
SOM_Base::SOM_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, ::CreateRuntimeParameter(), i_inputDataStruct, i_outputDataStruct)
{
}

/** デストラクタ */
SOM_Base::~SOM_Base()
{
}


//===========================
// 固有関数
//===========================
/** ニューロン数を取得する */
U32 SOM_Base::GetUnitCount()const
{
	const SOM_LayerData_Base* pLayerData = dynamic_cast<const SOM_LayerData_Base*>(&this->GetLayerData());
	if(pLayerData == NULL)
		return NULL;

	return pLayerData->GetUnitCount();
}
