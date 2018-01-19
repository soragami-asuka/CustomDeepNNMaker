//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A����������������
//======================================
#include"stdafx.h"

#include"SOM_FUNC.hpp"

#include"SOM_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** �R���X�g���N�^ */
SOM_Base::SOM_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, ::CreateRuntimeParameter(), i_inputDataStruct, i_outputDataStruct)
{
}

/** �f�X�g���N�^ */
SOM_Base::~SOM_Base()
{
}


//===========================
// �ŗL�֐�
//===========================
/** �j���[���������擾���� */
U32 SOM_Base::GetUnitCount()const
{
	const SOM_LayerData_Base* pLayerData = dynamic_cast<const SOM_LayerData_Base*>(&this->GetLayerData());
	if(pLayerData == NULL)
		return NULL;

	return pLayerData->GetUnitCount();
}
