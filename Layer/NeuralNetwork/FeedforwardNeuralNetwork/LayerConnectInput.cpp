//======================================
// ���C���[�Ԃ̐ڑ��ݒ�p�N���X.
// ���͐M���̑�p
//======================================
#include"stdafx.h"

#include"LayerConnect.h"
#include"FeedforwardNeuralNetwork_Base.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	
	/** �R���X�g���N�^ */
	LayerConnectInput::LayerConnectInput(class FeedforwardNeuralNetwork_Base& neuralNetwork)
		:	neuralNetwork	(neuralNetwork)
	{
	}
	/** �f�X�g���N�^ */
	LayerConnectInput::~LayerConnectInput()
	{
	}

	/** GUID���擾���� */
	Gravisbell::GUID LayerConnectInput::GetGUID()const
	{
		return this->neuralNetwork.GetInputGUID();
	}
	/** ���C���[��ʂ̎擾.
		ELayerKind �̑g�ݍ��킹. */
	unsigned int LayerConnectInput::GetLayerKind()const
	{
		return (this->neuralNetwork.GetLayerKind() & Gravisbell::Layer::LAYER_KIND_CALCTYPE) | Gravisbell::Layer::LAYER_KIND_SINGLE_OUTPUT;
	}

	/** �o�̓f�[�^�o�b�t�@���擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]
		@return �o�̓f�[�^�z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER LayerConnectInput::GetOutputBuffer()const
	{
		return NULL;
	}

	/** ���͌덷�o�b�t�@�̈ʒu����͌����C���[��GUID�w��Ŏ擾���� */
	S32 LayerConnectInput::GetDInputPositionByGUID(const Gravisbell::GUID& i_guid)const
	{
		return 0;
	}
	/** ���͌덷�o�b�t�@���ʒu�w��Ŏ擾���� */
	CONST_BATCH_BUFFER_POINTER LayerConnectInput::GetDInputBufferByNum(S32 num)const
	{
		if(this->lppOutputToLayer.empty())
			return NULL;
		return (*this->lppOutputToLayer.begin())->GetDInputBufferByNum(0);
	}

	/** ���C���[���X�g���쐬����.
		@param	i_lpLayerGUID	�ڑ����Ă���GUID�̃��X�g.���͕����Ɋm�F����. */
	ErrorCode LayerConnectInput::CreateLayerList(std::set<Gravisbell::GUID>& io_lpLayerGUID)const
	{
		io_lpLayerGUID.insert(this->GetGUID());

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** �v�Z�������X�g���쐬����.
		@param	i_lpLayerGUID		�S���C���[��GUID.
		@param	io_lpCalculateList	���Z���ɕ��ׂ�ꂽ�ڑ����X�g.
		@param	io_lpAddedList		�ڑ����X�g�ɓo�^�ς݂̃��C���[��GUID�ꗗ.
		@param	io_lpAddWaitList	�ǉ��ҋ@��Ԃ̐ڑ��N���X�̃��X�g. */
	ErrorCode LayerConnectInput::CreateCalculateList(const std::set<Gravisbell::GUID>& i_lpLayerGUID, std::list<ILayerConnect*>& io_lpCalculateList, std::set<Gravisbell::GUID>& io_lpAddedList, std::set<ILayerConnect*>& io_lpAddWaitList)const
	{
		// �擪�ɒǉ�
		io_lpCalculateList.insert(io_lpCalculateList.begin(), this);

		// �ǉ��ςɐݒ�
		io_lpAddedList.insert(this->GetGUID());

		// �ǉ��ҋ@��Ԃ̏ꍇ���� ���ǉ��ҋ@�ς݂ɂȂ邱�Ƃ͂��蓾�Ȃ�
		if(io_lpAddWaitList.count(this) > 0)
			io_lpAddWaitList.erase(this);

	}

	/** ���Z���O����.
		�ڑ��̊m�����s��. */
	ErrorCode LayerConnectInput::PreCalculate(void);

	/** ���Z���������s����. */
	ErrorCode LayerConnectInput::Calculate(void);
	/** �w�K�덷���v�Z����. */
	ErrorCode LayerConnectInput::CalculateLearnError(void);
	/** �w�K���������C���[�ɔ��f������.*/
	ErrorCode LayerConnectInput::ReflectionLearnError(void);


	/** ���C���[�ɓ��̓��C���[��ǉ�����. */
	ErrorCode LayerConnectInput::AddInputLayerToLayer(ILayerConnect* pInputFromLayer);
	/** ���C���[�Ƀo�C�p�X���C���[��ǉ�����.*/
	ErrorCode LayerConnectInput::AddBypassLayerToLayer(ILayerConnect* pInputFromLayer);

	/** ���C���[�̓��̓��C���[�ݒ�����Z�b�g����.
		@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
	ErrorCode LayerConnectInput::ResetInputLayer();
	/** ���C���[�̃o�C�p�X���C���[�ݒ�����Z�b�g����.
		@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
	ErrorCode LayerConnectInput::ResetBypassLayer();

	/** ���C���[�ɐڑ����Ă�����̓��C���[�̐����擾����.
		@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
	U32 LayerConnectInput::GetInputLayerCount()const;
	/** ���C���[�ɐڑ����Ă�����̓��C���[��GUID��ԍ��w��Ŏ擾����.
		@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
	ILayerConnect* LayerConnectInput::GetInputLayerByNum(U32 i_inputNum);

	/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[�̐����擾����. */
	U32 LayerConnectInput::GetBypassLayerCount()const;
	/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[��GUID��ԍ��w��Ŏ擾����.
		@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
	ILayerConnect* LayerConnectInput::GetBypassLayerByNum(U32 i_inputNum);


	/** �o�͐惌�C���[��ǉ����� */
	ErrorCode LayerConnectInput::AddOutputToLayer(ILayerConnect* pOutputToLayer);
	/** �o�͐惌�C���[���폜���� */
	ErrorCode LayerConnectInput::EraseOutputToLayer(const Gravisbell::GUID& guid);


	
}	// Gravisbell
}	// Layer
}	// NeuralNetwork