//======================================
// ��ݍ��݃j���[�����l�b�g���[�N�̌������C���[
//======================================
#include"stdafx.h"

#include"Convolution_FUNC.hpp"

#include"Convolution_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** �R���X�g���N�^ */
Convolution_Base::Convolution_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, ::CreateRuntimeParameter(), i_inputDataStruct, i_outputDataStruct)
{
}

/** �f�X�g���N�^ */
Convolution_Base::~Convolution_Base()
{
}