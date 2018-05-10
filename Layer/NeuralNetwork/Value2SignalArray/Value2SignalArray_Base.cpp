//======================================
// 出力信号分割レイヤー
//======================================
#include"stdafx.h"

#include"Value2SignalArray_FUNC.hpp"

#include"Value2SignalArray_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
Value2SignalArray_Base::Value2SignalArray_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, i_inputDataStruct, i_outputDataStruct)
{
}

/** デストラクタ */
Value2SignalArray_Base::~Value2SignalArray_Base()
{
}
