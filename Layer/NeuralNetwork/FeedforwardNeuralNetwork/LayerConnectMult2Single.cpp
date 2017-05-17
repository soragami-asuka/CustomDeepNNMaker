//======================================
// ���C���[�Ԃ̐ڑ��ݒ�p�N���X.
// �o�͐M���̑�p
//======================================
#include"stdafx.h"

#include"LayerConnectMult2Single.h"
#include"FeedforwardNeuralNetwork_Base.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	
	/** �R���X�g���N�^ */
	LayerConnectMult2Single::LayerConnectMult2Single(ILayerBase* pLayer, Gravisbell::SettingData::Standard::IData* pLearnSettingData)
		:	pLayer				(pLayer)
		,	pLayer_input		(dynamic_cast<INNMultInputLayer*>(pLayer))
		,	pLayer_output		(dynamic_cast<INNSingleOutputLayer*>(pLayer))
		,	pLearnSettingData	(pLearnSettingData)
	{
	}
	/** �f�X�g���N�^ */
	LayerConnectMult2Single::~LayerConnectMult2Single()
	{
		if(pLayer != NULL)
			delete pLayer;
		if(pLearnSettingData != NULL)
			delete pLearnSettingData;
	}

	/** GUID���擾���� */
	Gravisbell::GUID LayerConnectMult2Single::GetGUID()const
	{
		return this->pLayer->GetGUID();
	}
	/** ���C���[��ʂ̎擾.
		ELayerKind �̑g�ݍ��킹. */
	U32 LayerConnectMult2Single::GetLayerKind()const
	{
		return this->pLayer->GetLayerKind();
	}
	
	/** �w�K�ݒ�̃|�C���^���擾����.
		�擾�����f�[�^�𒼐ڏ��������邱�ƂŎ��̊w�K���[�v�ɔ��f����邪�ANULL���Ԃ��Ă��邱�Ƃ�����̂Œ���. */
	Gravisbell::SettingData::Standard::IData* LayerConnectMult2Single::GetLearnSettingData()
	{
		return this->pLearnSettingData;
	}

	/** �o�̓f�[�^�\�����擾����.
		@return	�o�̓f�[�^�\�� */
	IODataStruct LayerConnectMult2Single::GetOutputDataStruct()const
	{
		return this->pLayer_output->GetOutputDataStruct();
	}
	/** �o�̓f�[�^�o�b�t�@���擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]
		@return �o�̓f�[�^�z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER LayerConnectMult2Single::GetOutputBuffer()const
	{
		return this->pLayer_output->GetOutputBuffer();
	}

	/** ���͌덷�o�b�t�@�̈ʒu����͌����C���[��GUID�w��Ŏ擾���� */
	S32 LayerConnectMult2Single::GetDInputPositionByGUID(const Gravisbell::GUID& i_guid)const
	{
		for(U32 pos=0; pos<this->lppInputFromLayer.size(); pos++)
		{
			if(this->lppInputFromLayer[pos]->GetGUID() == i_guid)
				return pos;
		}

		return -1;
	}
	/** ���͌덷�o�b�t�@���ʒu�w��Ŏ擾���� */
	CONST_BATCH_BUFFER_POINTER LayerConnectMult2Single::GetDInputBufferByNum(S32 num)const
	{
		return this->pLayer_input->GetDInputBuffer(num);
	}

	/** ���C���[���X�g���쐬����.
		@param	i_lpLayerGUID	�ڑ����Ă���GUID�̃��X�g.���͕����Ɋm�F����. */
	ErrorCode LayerConnectMult2Single::CreateLayerList(std::set<Gravisbell::GUID>& io_lpLayerGUID)const
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
	ErrorCode LayerConnectMult2Single::CreateCalculateList(const std::set<Gravisbell::GUID>& i_lpLayerGUID, std::list<ILayerConnect*>& io_lpCalculateList, std::set<Gravisbell::GUID>& io_lpAddedList, std::set<ILayerConnect*>& io_lpAddWaitList)
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
	ErrorCode LayerConnectMult2Single::AddInputLayerToLayer(ILayerConnect* pInputFromLayer)
	{
		// ���͌����C���[�ɑ΂��Ď������o�͐�Ƃ��Đݒ�
		ErrorCode err = pInputFromLayer->AddOutputToLayer(this);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// ���͌����C���[�̃��X�g�ɒǉ�
		this->lppInputFromLayer.push_back(pInputFromLayer);

		return ErrorCode::ERROR_CODE_NONE;;
	}
	/** ���C���[�Ƀo�C�p�X���C���[��ǉ�����.*/
	ErrorCode LayerConnectMult2Single::AddBypassLayerToLayer(ILayerConnect* pInputFromLayer)
	{
		return ErrorCode::ERROR_CODE_ADDLAYER_UPPER_LIMIT;
	}


	/** ���C���[������̓��C���[���폜���� */
	ErrorCode LayerConnectMult2Single::EraseInputLayer(const Gravisbell::GUID& guid)
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
	ErrorCode LayerConnectMult2Single::EraseBypassLayer(const Gravisbell::GUID& guid)
	{
		return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;
	}


	/** ���C���[�̓��̓��C���[�ݒ�����Z�b�g����.
		@param	layerGUID	���Z�b�g���郌�C���[��GUID. */
	ErrorCode LayerConnectMult2Single::ResetInputLayer()
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
	ErrorCode LayerConnectMult2Single::ResetBypassLayer()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ���C���[�ɐڑ����Ă�����̓��C���[�̐����擾����.
		@param	i_layerGUID		�ڑ�����Ă��郌�C���[��GUID. */
	U32 LayerConnectMult2Single::GetInputLayerCount()const
	{
		return (U32)this->lppInputFromLayer.size();
	}
	/** ���C���[�ɐڑ����Ă�����̓��C���[��GUID��ԍ��w��Ŏ擾����.
		@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
	ILayerConnect* LayerConnectMult2Single::GetInputLayerByNum(U32 i_inputNum)
	{
		if(i_inputNum >= this->lppInputFromLayer.size())
			return NULL;

		return this->lppInputFromLayer[i_inputNum];
	}

	/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[�̐����擾����. */
	U32 LayerConnectMult2Single::GetBypassLayerCount()const
	{
		return 0;
	}
	/** ���C���[�ɐڑ����Ă���o�C�p�X���C���[��GUID��ԍ��w��Ŏ擾����.
		@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
	ILayerConnect* LayerConnectMult2Single::GetBypassLayerByNum(U32 i_inputNum)
	{
		return NULL;
	}

	
	//==========================================
	// �o�̓��C���[�֘A
	//==========================================

	/** �o�͐惌�C���[��ǉ����� */
	ErrorCode LayerConnectMult2Single::AddOutputToLayer(ILayerConnect* pOutputToLayer)
	{
		if(!this->lppOutputToLayer.empty())
			return ErrorCode::ERROR_CODE_ADDLAYER_UPPER_LIMIT;
		this->lppOutputToLayer.push_back(pOutputToLayer);

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** �o�͐惌�C���[���폜���� */
	ErrorCode LayerConnectMult2Single::EraseOutputToLayer(const Gravisbell::GUID& guid)
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
	U32 LayerConnectMult2Single::GetOutputToLayerCount()const
	{
		return (U32)this->lppOutputToLayer.size();
	}
	/** ���C���[�ɐڑ����Ă���o�͐惌�C���[��ԍ��w��Ŏ擾����.
		@param	i_inputNum		���C���[�ɐڑ����Ă��鉽�Ԗڂ̃��C���[���擾���邩�̎w��. */
	ILayerConnect* LayerConnectMult2Single::GetOutputToLayerByNum(U32 i_num)
	{
		if(i_num >= this->lppOutputToLayer.size())
			return NULL;

		return this->lppOutputToLayer[i_num].pLayer;
	}



	//=======================================
	// �ڑ��֘A
	//=======================================

	/** ���C���[�̐ڑ������� */
	ErrorCode LayerConnectMult2Single::Disconnect(void)
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

	//=======================================
	// ���Z�֘A
	//=======================================

	/** ���C���[�̏���������.
		�ڑ��󋵂͈ێ������܂܃��C���[�̒��g������������. */
	ErrorCode LayerConnectMult2Single::Initialize(void)
	{
		return this->pLayer->Initialize();
	}

	/** �ڑ��̊m�����s�� */
	ErrorCode LayerConnectMult2Single::EstablishmentConnection(void)
	{
		// ���͌����C���[���̊m�F
		if(this->lppInputFromLayer.empty())
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;
		if(this->lppInputFromLayer.size() != this->pLayer_input->GetInputDataCount())
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// ���͌����C���[�̃o�b�t�@�����m�F
		for(U32 inputNum=0; inputNum<this->lppInputFromLayer.size(); inputNum++)
		{
			if(this->lppInputFromLayer[inputNum]->GetOutputDataStruct().GetDataCount() != this->pLayer_input->GetInputBufferCount(inputNum))
				return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;
		}

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
	ErrorCode LayerConnectMult2Single::PreProcessLearn(unsigned int batchSize)
	{
		// ���͗p�̃o�b�t�@���쐬
		this->lppInputBuffer.resize(this->lppInputFromLayer.size());

		return this->pLayer->PreProcessLearn(batchSize);
	}
	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode LayerConnectMult2Single::PreProcessCalculate(unsigned int batchSize)
	{
		// ���͗p�̃o�b�t�@���쐬
		this->lppInputBuffer.resize(this->lppInputFromLayer.size());

		return this->pLayer->PreProcessCalculate(batchSize);
	}

	/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode LayerConnectMult2Single::PreProcessLearnLoop()
	{
		if(this->pLearnSettingData == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		return this->pLayer->PreProcessLearnLoop(*this->pLearnSettingData);
	}
	/** ���Z���[�v�̏���������.�f�[�^�Z�b�g�̉��Z�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode LayerConnectMult2Single::PreProcessCalculateLoop()
	{
		return this->pLayer->PreProcessCalculateLoop();
	}

	/** ���Z���������s����. */
	ErrorCode LayerConnectMult2Single::Calculate(void)
	{
		for(U32 inputNum=0; inputNum<this->lppInputFromLayer.size(); inputNum++)
		{
			this->lppInputBuffer[inputNum] = this->lppInputFromLayer[inputNum]->GetOutputBuffer();
		}

		return this->pLayer_input->Calculate(&this->lppInputBuffer[0]);
	}
	/** �w�K�덷���v�Z����. */
	ErrorCode LayerConnectMult2Single::Training(void)
	{
		return this->pLayer_output->Training(this->lppOutputToLayer[0].pLayer->GetDInputBufferByNum(this->lppOutputToLayer[0].position));
	}


}	// Gravisbell
}	// Layer
}	// NeuralNetwork