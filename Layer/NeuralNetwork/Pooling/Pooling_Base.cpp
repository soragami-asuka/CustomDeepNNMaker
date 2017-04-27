//======================================
// �����֐����C���[
//======================================
#include"stdafx.h"

#include"Pooling_FUNC.hpp"

#include"Pooling_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** �R���X�g���N�^ */
Pooling_Base::Pooling_Base(Gravisbell::GUID guid)
	:	INNLayer()
	,	guid				(guid)
	,	pLearnData			(NULL)
{
}

/** �f�X�g���N�^ */
Pooling_Base::~Pooling_Base()
{
	if(pLearnData != NULL)
		delete pLearnData;
}


//===========================
// ���C���[����
//===========================

/** ���C���[��ʂ̎擾.
	ELayerKind �̑g�ݍ��킹. */
U32 Pooling_Base::GetLayerKindBase(void)const
{
	return Gravisbell::Layer::ELayerKind::LAYER_KIND_NEURALNETWORK | Gravisbell::Layer::ELayerKind::LAYER_KIND_SINGLE_INPUT | Gravisbell::Layer::ELayerKind::LAYER_KIND_SINGLE_OUTPUT;
}

/** ���C���[�ŗL��GUID���擾���� */
Gravisbell::GUID Pooling_Base::GetGUID(void)const
{
	return this->guid;
}

/** ���C���[���ʃR�[�h���擾����.
	@param o_layerCode	�i�[��o�b�t�@
	@return ���������ꍇ0 */
Gravisbell::GUID Pooling_Base::GetLayerCode(void)const
{
	return this->GetLayerData().GetLayerCode();
}

/** �o�b�`�T�C�Y���擾����.
	@return �����ɉ��Z���s���o�b�`�̃T�C�Y */
unsigned int Pooling_Base::GetBatchSize()const
{
	return this->batchSize;
}

/** ���C���[�̐ݒ�����擾���� */
const SettingData::Standard::IData* Pooling_Base::GetLayerStructure()const
{
	return this->GetLayerData().GetLayerStructure();
}


//===========================
// ���̓��C���[�֘A
//===========================
/** ���̓f�[�^�\�����擾����.
	@return	���̓f�[�^�\�� */
IODataStruct Pooling_Base::GetInputDataStruct()const
{
	return this->GetLayerData().GetInputDataStruct();
}

/** ���̓o�b�t�@�����擾����. */
unsigned int Pooling_Base::GetInputBufferCount()const
{
	return this->GetLayerData().GetInputBufferCount();
}


//===========================
// ���C���[�ۑ�
//===========================
/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
unsigned int Pooling_Base::GetUseBufferByteCount()const
{
	return this->GetLayerData().GetUseBufferByteCount();
}


//===========================
// �o�̓��C���[�֘A
//===========================
/** �o�̓f�[�^�\�����擾���� */
IODataStruct Pooling_Base::GetOutputDataStruct()const
{
	return this->GetLayerData().GetOutputDataStruct();
}

/** �o�̓o�b�t�@�����擾���� */
unsigned int Pooling_Base::GetOutputBufferCount()const
{
	return this->GetLayerData().GetOutputBufferCount();
}


//===========================
// �ŗL�֐�
//===========================