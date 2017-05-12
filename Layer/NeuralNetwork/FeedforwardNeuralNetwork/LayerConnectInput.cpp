//======================================
// ���C���[�Ԃ̐ڑ��ݒ�p�N���X.
// ���͐M���̑�p
//======================================
#include"stdafx.h"

#include"LayerConnectInput.h"
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
	U32 LayerConnectInput::GetLayerKind()const
	{
		return (this->neuralNetwork.GetLayerKind() & Gravisbell::Layer::LAYER_KIND_CALCTYPE) | Gravisbell::Layer::LAYER_KIND_SINGLE_OUTPUT;
	}

	/** �w�K�ݒ�̃|�C���^���擾����.
		�擾�����f�[�^�𒼐ڏ��������邱�ƂŎ��̊w�K���[�v�ɔ��f����邪�ANULL���Ԃ��Ă��邱�Ƃ�����̂Œ���. */
	Gravisbell::SettingData::Standard::IData* LayerConnectInput::GetLearnSettingData()
	{
		return NULL;
	}

	/** �o�̓f�[�^�\�����擾����.
		@return	�o�̓f�[�^�\�� */
	IODataStruct LayerConnectInput::GetOutputDataStruct()const
	{
		return this->neuralNetwork.GetInputDataStruct();
	}
	/** �o�̓f�[�^�o�b�t�@���擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]
		@return �o�̓f�[�^�z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER LayerConnectInput::GetOutputBuffer()const
	{
		return this->neuralNetwork.GetInputBuffer();
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
		return (*this->lppOutputToLayer.begin()).pLayer->GetDInputBufferByNum((*this->lppOutputToLayer.begin()).position);
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
	ErrorCode LayerConnectInput::CreateCalculateList(const std::set<Gravisbell::GUID>& i_lpLayerGUID, std::list<ILayerConnect*>& io_lpCalculateList, std::set<Gravisbell::GUID>& io_lpAddedList, std::set<ILayerConnect*>& io_lpAddWaitList)
	{
		// �擪�ɒǉ�
		io_lpCalculateList.insert(io_lpCalculateList.begin(), this);

		// �ǉ��ςɐݒ�
		io_lpAddedList.insert(this->GetGUID());

		// �ǉ��ҋ@��Ԃ̏ꍇ���� ���ǉ��ҋ@�ς݂ɂȂ邱�Ƃ͂��蓾�Ȃ��̂�if����ʂ邱�Ƃ͂Ȃ�
		if(io_lpAddWaitList.count(this) > 0)
			io_lpAddWaitList.erase(this);

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���C���[�ɓ��̓��C���[��ǉ�����. */
	ErrorCode LayerConnectInput::AddInputLayerToLayer(ILayerConnect* pInputFromLayer)
	{
		return ErrorCode::ERROR_CODE_ADDLAYER_UPPER_LIMIT;
	}
	/** ���C���[�Ƀo�C�p�X���C���[��ǉ�����.*/
	ErrorCode LayerConnectInput::AddBypassLayerToLayer(ILayerConnect* pInputFromLayer)
	{
		return ErrorCode::ERROR_CODE_ADDLAYER_UPPER_LIMIT;
	}

	/** ���C���[������̓��C���[���폜���� */
	ErrorCode LayerConnectInput::EraseInputLayer(const Gravisbell::GUID& guid)
	{
		return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;
	}
	/** ���C���[����o�C�p�X���C���[���폜���� */
	ErrorCode LayerConnectInput::EraseBypassLayer(const Gravisbell::GUID& guid)
	{
		return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;
	}

	/** ���C���[�̓��̓��C���[�ݒ�����Z�b�g����.
		@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
	ErrorCode LayerConnectInput::ResetInputLayer()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ���C���[�̃o�C�p�X���C���[�ݒ�����Z�b�g����.
		@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
	ErrorCode LayerConnectInput::ResetBypassLayer()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ���C���[�ɐڑ����Ă�����̓��C���[�̐����擾����.
		@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
	U32 LayerConnectInput::GetInputLayerCount()const
	{
		return 0;
	}
	/** ���C���[�ɐڑ����Ă�����̓��C���[��ԍ��w��Ŏ擾����.
		@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
	ILayerConnect* LayerConnectInput::GetInputLayerByNum(U32 i_inputNum)
	{
		return NULL;
	}

	/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[�̐����擾����. */
	U32 LayerConnectInput::GetBypassLayerCount()const
	{
		return 0;
	}
	/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[��ԍ��w��Ŏ擾����.
		@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
	ILayerConnect* LayerConnectInput::GetBypassLayerByNum(U32 i_inputNum)
	{
		return NULL;
	}


	//==========================================
	// �o�̓��C���[�֘A
	//==========================================

	/** �o�͐惌�C���[��ǉ����� */
	ErrorCode LayerConnectInput::AddOutputToLayer(ILayerConnect* pOutputToLayer)
	{
		if(!this->lppOutputToLayer.empty())
			return ErrorCode::ERROR_CODE_ADDLAYER_UPPER_LIMIT;
		this->lppOutputToLayer.push_back(pOutputToLayer);

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** �o�͐惌�C���[���폜���� */
	ErrorCode LayerConnectInput::EraseOutputToLayer(const Gravisbell::GUID& guid)
	{
		auto it = this->lppOutputToLayer.begin();
		while(it != this->lppOutputToLayer.end())
		{
			if((*it).pLayer->GetGUID() == guid)
			{
				this->lppOutputToLayer.erase(it);
				return ErrorCode::ERROR_CODE_NONE;
			}
			it++;
		}
		return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;
	}
	
	/** ���C���[�ɐڑ����Ă���o�͐惌�C���[�̐����擾����.
		@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
	U32 LayerConnectInput::GetOutputToLayerCount()const
	{
		return (U32)this->lppOutputToLayer.size();
	}
	/** ���C���[�ɐڑ����Ă���o�͐惌�C���[��ԍ��w��Ŏ擾����.
		@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
	ILayerConnect* LayerConnectInput::GetOutputToLayerByNum(U32 i_num)
	{
		if(i_num >= this->lppOutputToLayer.size())
			return NULL;

		return this->lppOutputToLayer[i_num].pLayer;
	}

	
	/** ���C���[�̐ڑ������� */
	ErrorCode LayerConnectInput::Disconnect(void)
	{
		// �o�͐惌�C���[���玩�����폜
		for(auto it : this->lppOutputToLayer)
			it.pLayer->EraseInputLayer(this->GetGUID());

		// �o�͐惌�C���[��S�폜
		this->lppOutputToLayer.clear();

		this->ResetInputLayer();
		this->ResetBypassLayer();

		return ErrorCode::ERROR_CODE_NONE;
	}


	//=======================================
	// ���Z�֘A
	//=======================================
	
	/** ���C���[�̏���������.
		�ڑ��󋵂͈ێ������܂܃��C���[�̒��g������������. */
	ErrorCode LayerConnectInput::Initialize(void)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �ڑ��̊m�����s�� */
	ErrorCode LayerConnectInput::EstablishmentConnection(void)
	{
		// �o�͐惌�C���[���̊m�F
		if(this->lppOutputToLayer.empty())
			return ERROR_CODE_COMMON_NULL_REFERENCE;
		if(this->lppOutputToLayer.size() > 1)
			return ERROR_CODE_FRAUD_INPUT_COUNT;


		// �o�͐惌�C���[�̈ʒu��o�^
		auto it = this->lppOutputToLayer.begin();
		while(it != this->lppOutputToLayer.end())
		{
			it->position = it->pLayer->GetDInputPositionByGUID(this->GetGUID());

			it++;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ���Z�O���������s����.(�w�K�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
	ErrorCode LayerConnectInput::PreProcessLearn(unsigned int batchSize)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode LayerConnectInput::PreProcessCalculate(unsigned int batchSize)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode LayerConnectInput::PreProcessLearnLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ���Z���[�v�̏���������.�f�[�^�Z�b�g�̉��Z�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode LayerConnectInput::PreProcessCalculateLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����. */
	ErrorCode LayerConnectInput::Calculate(void)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}
	/** �w�K���������s����. */
	ErrorCode LayerConnectInput::Training(void)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}

	//==================================
	// �ۑ��֘A
	//==================================
	/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
	U32 LayerConnectInput::GetUseBufferByteCount()const
	{
		return 0;
	}

	/** ���C���[���o�b�t�@�ɏ�������.
		@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
		@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
	S32 LayerConnectInput::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		return -1;
	}

}	// Gravisbell
}	// Layer
}	// NeuralNetwork