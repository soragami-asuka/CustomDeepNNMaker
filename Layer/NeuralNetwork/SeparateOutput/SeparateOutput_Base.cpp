//======================================
// 出力信号分割レイヤー
//======================================
#include"stdafx.h"

#include"SeparateOutput_FUNC.hpp"

#include"SeparateOutput_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
SeparateOutput_Base::SeparateOutput_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2MultLayerBase(guid, ::CreateLearningSetting(), i_inputDataStruct, i_outputDataStruct)
{
}

/** デストラクタ */
SeparateOutput_Base::~SeparateOutput_Base()
{
}


//===========================
// 出力レイヤー関連
//===========================
/** 出力データの出力先レイヤー数. */
U32 SeparateOutput_Base::GetOutputToLayerCount()const
{
	const SeparateOutput_LayerData_Base* pLayerData = dynamic_cast<const SeparateOutput_LayerData_Base*>(&this->GetLayerData());
	if(pLayerData == NULL)
		return 0;

	return pLayerData->GetOutputToLayerCount();
}