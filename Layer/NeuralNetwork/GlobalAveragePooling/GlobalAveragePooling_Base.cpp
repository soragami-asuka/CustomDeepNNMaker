//======================================
// �����֐����C���[
//======================================
#include"stdafx.h"

#include"GlobalAveragePooling_FUNC.hpp"

#include"GlobalAveragePooling_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** �R���X�g���N�^ */
GlobalAveragePooling_Base::GlobalAveragePooling_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, ::CreateLearningSetting(), i_inputDataStruct, i_outputDataStruct)
{
}

/** �f�X�g���N�^ */
GlobalAveragePooling_Base::~GlobalAveragePooling_Base()
{
}
