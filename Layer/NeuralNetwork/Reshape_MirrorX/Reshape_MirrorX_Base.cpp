//======================================
// 出力信号分割レイヤー
//======================================
#include"stdafx.h"

#include"Reshape_MirrorX_FUNC.hpp"

#include"Reshape_MirrorX_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
Reshape_MirrorX_Base::Reshape_MirrorX_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, i_inputDataStruct, i_outputDataStruct)
{
}

/** デストラクタ */
Reshape_MirrorX_Base::~Reshape_MirrorX_Base()
{
}
