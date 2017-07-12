//======================================
// ���C���[�Ԃ̐ڑ��ݒ�p�N���X.
// �o�͐M���̑�p
//======================================
#include"stdafx.h"

#include"LayerConnectSingle2Single.h"
#include"FeedforwardNeuralNetwork_Base.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	
	/** �R���X�g���N�^ */
	LayerConnectSingle2Single::LayerConnectSingle2Single(class FeedforwardNeuralNetwork_Base& neuralNetwork, ILayerBase* pLayer, Gravisbell::SettingData::Standard::IData* pRuntimeParameter)
		:	neuralNetwork		(neuralNetwork)
		,	pLayer				(pLayer)
		,	pLayer_io			(dynamic_cast<INNSingle2SingleLayer*>(pLayer))
		,	pRuntimeParameter	(pRuntimeParameter)
		,	dInputBufferID		(INVALID_DINPUTBUFFER_ID)
	{
	}
	/** �f�X�g���N�^ */
	LayerConnectSingle2Single::~LayerConnectSingle2Single()
	{
		if(pLayer != NULL)
			delete pLayer;
		if(pRuntimeParameter != NULL)
			delete pRuntimeParameter;
	}

	/** GUID���擾���� */
	Gravisbell::GUID LayerConnectSingle2Single::GetGUID()const
	{
		return this->pLayer->GetGUID();
	}
	/** ���C���[��ʂ̎擾.
		ELayerKind �̑g�ݍ��킹. */
	U32 LayerConnectSingle2Single::GetLayerKind()const
	{
		return this->pLayer->GetLayerKind();
	}
	
	/** �w�K�ݒ�̃|�C���^���擾����.
		�擾�����f�[�^�𒼐ڏ��������邱�ƂŎ��̊w�K���[�v�ɔ��f����邪�ANULL���Ԃ��Ă��邱�Ƃ�����̂Œ���. */
	Gravisbell::SettingData::Standard::IData* LayerConnectSingle2Single::GetRuntimeParameter()
	{
		return this->pRuntimeParameter;
	}

	/** �o�̓f�[�^�\�����擾����.
		@return	�o�̓f�[�^�\�� */
	IODataStruct LayerConnectSingle2Single::GetOutputDataStruct()const
	{
		return this->pLayer_io->GetOutputDataStruct();
	}
	/** �o�̓f�[�^�o�b�t�@���擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]
		@return �o�̓f�[�^�z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER LayerConnectSingle2Single::GetOutputBuffer()const
	{
		return this->pLayer_io->GetOutputBuffer();
	}

	/** ���͌덷�o�b�t�@�̈ʒu����͌����C���[��GUID�w��Ŏ擾���� */
	S32 LayerConnectSingle2Single::GetDInputPositionByGUID(const Gravisbell::GUID& i_guid)const
	{
		for(U32 pos=0; pos<this->lppInputFromLayer.size(); pos++)
		{
			if(this->lppInputFromLayer[pos]->GetGUID() == i_guid)
				return pos;
		}

		return -1;
	}
	/** ���͌덷�o�b�t�@���ʒu�w��Ŏ擾���� */
	CONST_BATCH_BUFFER_POINTER LayerConnectSingle2Single::GetDInputBufferByNum(S32 num)const
	{
		return neuralNetwork.GetDInputBuffer(this->GetDInputBufferID(0));
	}

	/** ���C���[���X�g���쐬����.
		@param	i_lpLayerGUID	�ڑ����Ă���GUID�̃��X�g.���͕����Ɋm�F����. */
	ErrorCode LayerConnectSingle2Single::CreateLayerList(std::set<Gravisbell::GUID>& io_lpLayerGUID)const
	{
		if(io_lpLayerGUID.count(this->GetGUID()))
			return ErrorCode::ERROR_CODE_NONE;

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
	ErrorCode LayerConnectSingle2Single::CreateCalculateList(const std::set<Gravisbell::GUID>& i_lpLayerGUID, std::list<ILayerConnect*>& io_lpCalculateList, std::set<Gravisbell::GUID>& io_lpAddedList, std::set<ILayerConnect*>& io_lpAddWaitList)
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
	ErrorCode LayerConnectSingle2Single::AddInputLayerToLayer(ILayerConnect* pInputFromLayer)
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
	ErrorCode LayerConnectSingle2Single::AddBypassLayerToLayer(ILayerConnect* pInputFromLayer)
	{
		return ErrorCode::ERROR_CODE_ADDLAYER_UPPER_LIMIT;
	}


	/** ���C���[������̓��C���[���폜���� */
	ErrorCode LayerConnectSingle2Single::EraseInputLayer(const Gravisbell::GUID& guid)
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
	ErrorCode LayerConnectSingle2Single::EraseBypassLayer(const Gravisbell::GUID& guid)
	{
		return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;
	}


	/** ���C���[�̓��̓��C���[�ݒ�����Z�b�g����.
		@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
	ErrorCode LayerConnectSingle2Single::ResetInputLayer()
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
	ErrorCode LayerConnectSingle2Single::ResetBypassLayer()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ���C���[�ɐڑ����Ă�����̓��C���[�̐����擾����.
		@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
	U32 LayerConnectSingle2Single::GetInputLayerCount()const
	{
		return (U32)this->lppInputFromLayer.size();
	}
	/** ���C���[�ɐڑ����Ă�����̓��C���[��GUID��ԍ��w��Ŏ擾����.
		@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
	ILayerConnect* LayerConnectSingle2Single::GetInputLayerByNum(U32 i_inputNum)
	{
		if(i_inputNum >= this->lppInputFromLayer.size())
			return NULL;

		return this->lppInputFromLayer[i_inputNum];
	}

	/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[�̐����擾����. */
	U32 LayerConnectSingle2Single::GetBypassLayerCount()const
	{
		return 0;
	}
	/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[��GUID��ԍ��w��Ŏ擾����.
		@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
	ILayerConnect* LayerConnectSingle2Single::GetBypassLayerByNum(U32 i_inputNum)
	{
		return NULL;
	}

	
	//==========================================
	// �o�̓��C���[�֘A
	//==========================================

	/** �o�͐惌�C���[��ǉ����� */
	ErrorCode LayerConnectSingle2Single::AddOutputToLayer(ILayerConnect* pOutputToLayer)
	{
		if(!this->lppOutputToLayer.empty())
			return ErrorCode::ERROR_CODE_ADDLAYER_UPPER_LIMIT;
		this->lppOutputToLayer.push_back(pOutputToLayer);

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** �o�͐惌�C���[���폜���� */
	ErrorCode LayerConnectSingle2Single::EraseOutputToLayer(const Gravisbell::GUID& guid)
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
	U32 LayerConnectSingle2Single::GetOutputToLayerCount()const
	{
		return (U32)this->lppOutputToLayer.size();
	}
	/** ���C���[�ɐڑ����Ă���o�͐惌�C���[��ԍ��w��Ŏ擾����.
		@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
	ILayerConnect* LayerConnectSingle2Single::GetOutputToLayerByNum(U32 i_num)
	{
		if(i_num >= this->lppOutputToLayer.size())
			return NULL;

		return this->lppOutputToLayer[i_num].pLayer;
	}



	//=======================================
	// �ڑ��֘A
	//=======================================

	/** ���C���[�̐ڑ������� */
	ErrorCode LayerConnectSingle2Single::Disconnect(void)
	{
		// �o�͐惌�C���[���玩�����폜
		for(auto it : this->lppOutputToLayer)
			it.pLayer->EraseInputLayer(this->GetGUID());

		// �o�͐惌�C���[��S�폜
		this->lppOutputToLayer.clear();

		// ���g�̓��̓��C���[/�o�C�p�X���C���[���폜
		this->ResetInputLayer();
		this->ResetBypassLayer();

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���C���[�Ŏg�p������͌덷�o�b�t�@��ID���擾����
		@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
	S32 LayerConnectSingle2Single::GetDInputBufferID(U32 i_inputNum)const
	{
		return this->dInputBufferID;
	}
	/** ���C���[�Ŏg�p������͌덷�o�b�t�@��ID���擾����
		@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
	ErrorCode LayerConnectSingle2Single::SetDInputBufferID(U32 i_inputNum, S32 i_DInputBufferID)
	{
		this->dInputBufferID = i_DInputBufferID;

		return ErrorCode::ERROR_CODE_NONE;
	}


	//=======================================
	// ���Z�֘A
	//=======================================

	/** ���C���[�̏���������.
		�ڑ��󋵂͈ێ������܂܃��C���[�̒��g������������. */
	ErrorCode LayerConnectSingle2Single::Initialize(void)
	{
		return this->pLayer->Initialize();
	}

	/** �ڑ��̊m�����s�� */
	ErrorCode LayerConnectSingle2Single::EstablishmentConnection(void)
	{
		// ���͌����C���[���̊m�F
		if(this->lppInputFromLayer.empty())
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;
		if(this->lppInputFromLayer.size() > 1)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// ���͌����C���[�̃o�b�t�@�����m�F
		if(this->lppInputFromLayer[0]->GetOutputDataStruct().GetDataCount() != this->pLayer_io->GetInputBufferCount())
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// �o�͐惌�C���[���̊m�F
		if(this->lppOutputToLayer.empty())
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;
		if(this->lppOutputToLayer.size() > 1)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// �o�͐惌�C���[�̈ʒu��o�^
		this->lppOutputToLayer[0].position = this->lppOutputToLayer[0].pLayer->GetDInputPositionByGUID(this->GetGUID());

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ���Z�O���������s����.(�w�K�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
	ErrorCode LayerConnectSingle2Single::PreProcessLearn(unsigned int batchSize)
	{
		return this->pLayer->PreProcessLearn(batchSize);
	}
	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode LayerConnectSingle2Single::PreProcessCalculate(unsigned int batchSize)
	{
		return this->pLayer->PreProcessCalculate(batchSize);
	}
	

	/** �������[�v�̏���������.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode LayerConnectSingle2Single::PreProcessLoop()
	{
		return this->pLayer->PreProcessLoop();
	}

	/** ���Z���������s����. */
	ErrorCode LayerConnectSingle2Single::Calculate(void)
	{
		return this->pLayer_io->Calculate(lppInputFromLayer[0]->GetOutputBuffer());
	}
	/** �w�K�덷���v�Z����. */
	ErrorCode LayerConnectSingle2Single::CalculateDInput(void)
	{
		if(this->GetDInputBufferID(0) < 0)
		{
			return this->pLayer_io->CalculateDInput(
				this->neuralNetwork.GetDInputBuffer(),
				this->lppOutputToLayer[0].pLayer->GetDInputBufferByNum(this->lppOutputToLayer[0].position) );
		}
		else
		{
			return this->pLayer_io->CalculateDInput(
				this->neuralNetwork.GetDInputBuffer(this->GetDInputBufferID(0)),
				this->lppOutputToLayer[0].pLayer->GetDInputBufferByNum(this->lppOutputToLayer[0].position) );
		}
	}
	/** �w�K�덷���v�Z����. */
	ErrorCode LayerConnectSingle2Single::Training(void)
	{
		if(this->GetDInputBufferID(0) < 0)
		{
			return this->pLayer_io->Training(
				this->neuralNetwork.GetDInputBuffer(),
				this->lppOutputToLayer[0].pLayer->GetDInputBufferByNum(this->lppOutputToLayer[0].position) );
		}
		else
		{
			return this->pLayer_io->Training(
				this->neuralNetwork.GetDInputBuffer(this->GetDInputBufferID(0)),
				this->lppOutputToLayer[0].pLayer->GetDInputBufferByNum(this->lppOutputToLayer[0].position) );
		}
	}


}	// Gravisbell
}	// Layer
}	// NeuralNetwork