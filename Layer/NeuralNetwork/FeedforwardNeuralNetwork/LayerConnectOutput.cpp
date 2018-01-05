//======================================
// ���C���[�Ԃ̐ڑ��ݒ�p�N���X.
// �o�͐M���̑�p
//======================================
#include"stdafx.h"

#include"LayerConnectOutput.h"
#include"FeedforwardNeuralNetwork_Base.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	
	/** �R���X�g���N�^ */
	LayerConnectOutput::LayerConnectOutput(class FeedforwardNeuralNetwork_Base& neuralNetwork)
		:	neuralNetwork	(neuralNetwork)
		,	dInputBufferID	(INVALID_DINPUTBUFFER_ID)	/**< ���͌덷�o�b�t�@ID */
	{
	}
	/** �f�X�g���N�^ */
	LayerConnectOutput::~LayerConnectOutput()
	{
	}

	/** GUID���擾���� */
	Gravisbell::GUID LayerConnectOutput::GetGUID()const
	{
		return this->neuralNetwork.GetGUID();
	}
	/** ���C���[��ʂ̎擾.
		ELayerKind �̑g�ݍ��킹. */
	U32 LayerConnectOutput::GetLayerKind()const
	{
		return (this->neuralNetwork.GetLayerKind() & Gravisbell::Layer::LAYER_KIND_CALCTYPE) | Gravisbell::Layer::LAYER_KIND_SINGLE_INPUT;
	}


	//====================================
	// ���s���ݒ�
	//====================================
	/** ���s���ݒ���擾����. */
	const SettingData::Standard::IData* LayerConnectOutput::GetRuntimeParameter()const
	{
		return NULL;
	}

	/** ���s���ݒ��ݒ肷��.
		int�^�Afloat�^�Aenum�^���Ώ�.
		@param	i_dataID	�ݒ肷��l��ID.
		@param	i_param		�ݒ肷��l. */
	ErrorCode LayerConnectOutput::SetRuntimeParameter(const wchar_t* i_dataID, S32 i_param)
	{
		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	}
	/** ���s���ݒ��ݒ肷��.
		int�^�Afloat�^���Ώ�.
		@param	i_dataID	�ݒ肷��l��ID.
		@param	i_param		�ݒ肷��l. */
	ErrorCode LayerConnectOutput::SetRuntimeParameter(const wchar_t* i_dataID, F32 i_param)
	{
		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	}
	/** ���s���ݒ��ݒ肷��.
		bool�^���Ώ�.
		@param	i_dataID	�ݒ肷��l��ID.
		@param	i_param		�ݒ肷��l. */
	ErrorCode LayerConnectOutput::SetRuntimeParameter(const wchar_t* i_dataID, bool i_param)
	{
		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	}
	/** ���s���ݒ��ݒ肷��.
		string�^���Ώ�.
		@param	i_dataID	�ݒ肷��l��ID.
		@param	i_param		�ݒ肷��l. */
	ErrorCode LayerConnectOutput::SetRuntimeParameter(const wchar_t* i_dataID, const wchar_t* i_param)
	{
		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	}


		
	//====================================
	// ���o�̓f�[�^�\��
	//====================================
	/** �o�̓f�[�^�\�����擾����.
		@return	�o�̓f�[�^�\�� */
	IODataStruct LayerConnectOutput::GetOutputDataStruct()const
	{
		if(this->lppInputFromLayer.empty())
			return IODataStruct();

		return this->lppInputFromLayer[0]->GetOutputDataStruct();
	}
	/** �o�̓f�[�^�o�b�t�@���擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]
		@return �o�̓f�[�^�z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER LayerConnectOutput::GetOutputBuffer_d()const
	{
		if(this->lppInputFromLayer.empty())
			return NULL;

		return this->lppInputFromLayer[0]->GetOutputBuffer_d();
	}

	/** ���͌덷�o�b�t�@�̈ʒu����͌����C���[��GUID�w��Ŏ擾���� */
	S32 LayerConnectOutput::GetDInputPositionByGUID(const Gravisbell::GUID& i_guid)const
	{
		return 0;
	}
	/** ���͌덷�o�b�t�@���ʒu�w��Ŏ擾���� */
	CONST_BATCH_BUFFER_POINTER LayerConnectOutput::GetDInputBufferByNum_d(S32 num)const
	{
		return this->neuralNetwork.GetDOutputBuffer_d();
	}

	/** ���C���[���X�g���쐬����.
		@param	i_lpLayerGUID	�ڑ����Ă���GUID�̃��X�g.���͕����Ɋm�F����. */
	ErrorCode LayerConnectOutput::CreateLayerList(std::set<Gravisbell::GUID>& io_lpLayerGUID)const
	{
		io_lpLayerGUID.insert(this->GetGUID());
		for(auto pInputFromLayer : this->lppInputFromLayer)
			pInputFromLayer->CreateLayerList(io_lpLayerGUID);

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** �v�Z�������X�g���쐬����.
		@param	i_lpLayerGUID		�S���C���[��GUID.
		@param	io_lpCalculateList	���Z���ɕ��ׂ�ꂽ�ڑ����X�g.
		@param	io_lpAddedList		�ڑ����X�g�ɓo�^�ς݂̃��C���[��GUID�ꗗ.
		@param	io_lpAddWaitList	�ǉ��ҋ@��Ԃ̐ڑ��N���X�̃��X�g. */
	ErrorCode LayerConnectOutput::CreateCalculateList(const std::set<Gravisbell::GUID>& i_lpLayerGUID, std::list<ILayerConnect*>& io_lpCalculateList, std::set<Gravisbell::GUID>& io_lpAddedList, std::set<ILayerConnect*>& io_lpAddWaitList)
	{
		// �ǉ�����
		{
			// �擪�ɒǉ�
			io_lpCalculateList.insert(io_lpCalculateList.begin(), this);

			// �ǉ��ςɐݒ�
			io_lpAddedList.insert(this->GetGUID());

			// �ǉ��ҋ@��Ԃ̏ꍇ���� ���ǉ��ҋ@�ς݂ɂȂ邱�Ƃ͂��蓾�Ȃ��̂�if����ʂ邱�Ƃ͂Ȃ����O�̂���
			if(io_lpAddWaitList.count(this) > 0)
				io_lpAddWaitList.erase(this);
		}

		// ���͌����C���[��ǉ�
		for(auto pInputFromLayer : this->lppInputFromLayer)
		{
			if(pInputFromLayer->GetLayerKind() & Gravisbell::Layer::LAYER_KIND_MULT_OUTPUT)
			{
				// �����o�̓��C���[�̏ꍇ�͈�U�ۗ�
				io_lpAddWaitList.insert(pInputFromLayer);
			}
			else
			{
				// �P�Əo�̓��C���[�̏ꍇ�͏��������s
				ErrorCode errCode = pInputFromLayer->CreateCalculateList(i_lpLayerGUID, io_lpCalculateList, io_lpAddedList, io_lpAddWaitList);
				if(errCode != ErrorCode::ERROR_CODE_NONE)
					return errCode;
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���C���[�ɓ��̓��C���[��ǉ�����. */
	ErrorCode LayerConnectOutput::AddInputLayerToLayer(ILayerConnect* pInputFromLayer)
	{
		if(!this->lppInputFromLayer.empty())
			return ErrorCode::ERROR_CODE_ADDLAYER_UPPER_LIMIT;

		// ���͌����C���[�ɑ΂��Ď������o�͐�Ƃ��Đݒ�
		ErrorCode err = pInputFromLayer->AddOutputToLayer(this);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// ���͌����C���[�̃��X�g�ɒǉ�
		this->lppInputFromLayer.push_back(pInputFromLayer);

		return ErrorCode::ERROR_CODE_NONE;;
	}
	/** ���C���[�Ƀo�C�p�X���C���[��ǉ�����.*/
	ErrorCode LayerConnectOutput::AddBypassLayerToLayer(ILayerConnect* pInputFromLayer)
	{
		return ErrorCode::ERROR_CODE_ADDLAYER_UPPER_LIMIT;
	}


	/** ���C���[������̓��C���[���폜���� */
	ErrorCode LayerConnectOutput::EraseInputLayer(const Gravisbell::GUID& guid)
	{
		auto it = this->lppInputFromLayer.begin();
		while(it != this->lppInputFromLayer.end())
		{
			if((*it)->GetGUID() == guid)
			{
				it = this->lppInputFromLayer.erase(it);
				return ErrorCode::ERROR_CODE_NONE;
			}
			else
			{
				it++;
			}
		}

		return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;
	}
	/** ���C���[����o�C�p�X���C���[���폜���� */
	ErrorCode LayerConnectOutput::EraseBypassLayer(const Gravisbell::GUID& guid)
	{
		return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;
	}


	/** ���C���[�̓��̓��C���[�ݒ�����Z�b�g����.
		@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
	ErrorCode LayerConnectOutput::ResetInputLayer()
	{
		auto it = this->lppInputFromLayer.begin();
		while(it != this->lppInputFromLayer.end())
		{
			// ���͌�����o�͐���폜
			(*it)->EraseOutputToLayer(this->GetGUID());

			// ���͌����C���[���폜
			it = this->lppInputFromLayer.erase(it);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ���C���[�̃o�C�p�X���C���[�ݒ�����Z�b�g����.
		@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
	ErrorCode LayerConnectOutput::ResetBypassLayer()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ���C���[�ɐڑ����Ă�����̓��C���[�̐����擾����.
		@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
	U32 LayerConnectOutput::GetInputLayerCount()const
	{
		return (U32)this->lppInputFromLayer.size();
	}
	/** ���C���[�ɐڑ����Ă�����̓��C���[��GUID��ԍ��w��Ŏ擾����.
		@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
	ILayerConnect* LayerConnectOutput::GetInputLayerByNum(U32 i_inputNum)
	{
		if(i_inputNum >= this->lppInputFromLayer.size())
			return NULL;

		return this->lppInputFromLayer[i_inputNum];
	}

	/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[�̐����擾����. */
	U32 LayerConnectOutput::GetBypassLayerCount()const
	{
		return 0;
	}
	/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[��GUID��ԍ��w��Ŏ擾����.
		@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
	ILayerConnect* LayerConnectOutput::GetBypassLayerByNum(U32 i_inputNum)
	{
		return NULL;
	}
	

	//==========================================
	// �o�̓��C���[�֘A
	//==========================================

	/** �o�͐惌�C���[��ǉ����� */
	ErrorCode LayerConnectOutput::AddOutputToLayer(ILayerConnect* pOutputToLayer)
	{
		return ErrorCode::ERROR_CODE_ADDLAYER_UPPER_LIMIT;
	}
	/** �o�͐惌�C���[���폜���� */
	ErrorCode LayerConnectOutput::EraseOutputToLayer(const Gravisbell::GUID& guid)
	{
		return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;
	}
	
	/** ���C���[�ɐڑ����Ă���o�͐惌�C���[�̐����擾����.
		@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
	U32 LayerConnectOutput::GetOutputToLayerCount()const
	{
		return 0;
	}
	/** ���C���[�ɐڑ����Ă���o�͐惌�C���[��ԍ��w��Ŏ擾����.
		@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
	ILayerConnect* LayerConnectOutput::GetOutputToLayerByNum(U32 i_num)
	{
		return NULL;
	}



	//=======================================
	// �ڑ��֘A
	//=======================================
	/** ���C���[�̐ڑ������� */
	ErrorCode LayerConnectOutput::Disconnect(void)
	{
		this->ResetInputLayer();
		this->ResetBypassLayer();

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ���C���[�Ŏg�p����o�̓o�b�t�@��ID��o�^���� */
	ErrorCode LayerConnectOutput::SetOutputBufferID(S32 i_outputBufferID)
	{
		// �o�̓��C���[�̏o�̓o�b�t�@�͒��O�̃��C���[�̏o�̓o�b�t�@�Ɠ���̂��߁A�o�^�s�v

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ���C���[�Ŏg�p������͌덷�o�b�t�@��ID���擾����
		@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
	S32 LayerConnectOutput::GetDInputBufferID(U32 i_inputNum)const
	{
		return this->dInputBufferID;
	}
	/** ���C���[�Ŏg�p������͌덷�o�b�t�@��ID���擾����
		@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
	ErrorCode LayerConnectOutput::SetDInputBufferID(U32 i_inputNum, S32 i_DInputBufferID)
	{
		this->dInputBufferID = i_DInputBufferID;

		return ErrorCode::ERROR_CODE_NONE;
	}


	//=======================================
	// ���Z�֘A
	//=======================================
	/** ���C���[�̏���������.
		�ڑ��󋵂͈ێ������܂܃��C���[�̒��g������������. */
	ErrorCode LayerConnectOutput::Initialize(void)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �ڑ��̊m�����s�� */
	ErrorCode LayerConnectOutput::EstablishmentConnection(void)
	{
		// ���͌����C���[���̊m�F
		if(this->lppInputFromLayer.empty())
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;
		if(this->lppInputFromLayer.size() > 1)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ���Z�O���������s����.(�w�K�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
	ErrorCode LayerConnectOutput::PreProcessLearn(unsigned int batchSize)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode LayerConnectOutput::PreProcessCalculate(unsigned int batchSize)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}
	
	/** �������[�v�̏���������.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode LayerConnectOutput::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����. */
	ErrorCode LayerConnectOutput::Calculate(void)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}
	/** �w�K���������s����. */
	ErrorCode LayerConnectOutput::CalculateDInput(void)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}
	/** �w�K���������s����. */
	ErrorCode LayerConnectOutput::Training(void)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}

}	// Gravisbell
}	// Layer
}	// NeuralNetwork