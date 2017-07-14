//============================================
// �S�Ẵj���[�����l�b�g���[�N�n���C���[�̃x�[�X�ƂȂ鋤�ʏ���
// ���O�Ŏ����\�ł���ΕK���p������K�v�͂Ȃ�
//============================================
#ifndef __GRAVISBELL_LAYER_NEURALNETWORK_CLAYREBASE_H__
#define __GRAVISBELL_LAYER_NEURALNETWORK_CLAYREBASE_H__


#include<Layer/NeuralNetwork/INNSingle2SingleLayer.h>
#include<Layer/NeuralNetwork/INNSingle2MultLayer.h>
#include<Layer/NeuralNetwork/INNMult2SingleLayer.h>


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	template<typename IOLayer>
	class CLayerBase : public IOLayer
	{
	protected:
		enum ProcessType
		{
			PROCESSTYPE_NONE,
			PROCESSTYPE_CALCULATE,
			PROCESSTYPE_LEARN,

			PROCESSTYPE_COUNT
		};

	private:
		Gravisbell::GUID guid;	/**< ���C���[���ʗp��GUID */

		ProcessType processType;
		U32 batchSize;	/**< �o�b�`�T�C�Y */

	public:
		/** �R���X�g���N�^ */
		CLayerBase(Gravisbell::GUID guid)
			:	guid						(guid)
			,	processType					(PROCESSTYPE_CALCULATE)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~CLayerBase()
		{
		}


		//===========================
		// ���C���[����
		//===========================
	public:

		/** ���C���[�ŗL��GUID���擾���� */
		Gravisbell::GUID GetGUID(void)const
		{
			return this->guid;
		}

		/** ���C���[�̎�ގ��ʃR�[�h���擾����.
			@param o_layerCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		Gravisbell::GUID GetLayerCode(void)const
		{
			return this->GetLayerData().GetLayerCode();
		}

		/** ���C���[�̐ݒ�����擾���� */
		const SettingData::Standard::IData* GetLayerStructure()const
		{
			return this->GetLayerData().GetLayerStructure();
		}

		//===========================
		// ���C���[�����֘A
		//===========================
		/** ���Z�O���������s����.(�w�K�p)
			@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
			NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
			���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
		ErrorCode PreProcessLearn(U32 batchSize)
		{
			this->batchSize = batchSize;
			this->processType = PROCESSTYPE_LEARN;

			return PreProcessLearn();
		}
		virtual ErrorCode PreProcessLearn() = 0;

		/** ���Z�O���������s����.(���Z�p)
			@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
			NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		ErrorCode PreProcessCalculate(U32 batchSize)
		{
			this->batchSize = batchSize;
			this->processType = PROCESSTYPE_CALCULATE;

			return PreProcessCalculate();
		}
		virtual ErrorCode PreProcessCalculate() = 0;


		/** �o�b�`�T�C�Y���擾����.
			@return �����ɉ��Z���s���o�b�`�̃T�C�Y */
		U32 GetBatchSize()const
		{
			return this->batchSize;
		}

		/** ���Z��ʂ��擾���� */
		ProcessType GetProcessType()const
		{
			return this->processType;
		}


		//===========================
		// ���C���[�f�[�^�֘A
		//===========================
	public:
		/** ���C���[�f�[�^���擾���� */
		virtual ILayerData& GetLayerData() = 0;
		virtual const ILayerData& GetLayerData()const = 0;
	};


	template<typename IOLayer, typename RuntimeParameter>
	class CLayerBaseRuntimeParameter  : public CLayerBase<IOLayer>
	{
	private:
		SettingData::Standard::IData* pRuntimeParameter;
		bool onUpdateRuntimeParameter;	// ���s���p�����[�^���X�V�����t���O

		RuntimeParameter runtimeParameter;

	protected:
		/** �R���X�g���N�^ */
		CLayerBaseRuntimeParameter(Gravisbell::GUID guid, SettingData::Standard::IData* i_pRuntimeParameter)
			:	CLayerBase					(guid)
			,	pRuntimeParameter			(i_pRuntimeParameter)
			,	onUpdateRuntimeParameter	(false)
			,	runtimeParameter			()
		{
		}
		/** �f�X�g���N�^ */
		virtual ~CLayerBaseRuntimeParameter()
		{
			if(pRuntimeParameter)
				delete pRuntimeParameter;
		}
		

	protected:
		/** ���s���ݒ���\���̂Ŏ擾���� */
		const RuntimeParameter& GetRuntimeParameterByStructure()
		{
			if(this->onUpdateRuntimeParameter)
			{
				this->pRuntimeParameter->WriteToStruct((BYTE*)&runtimeParameter);
				this->onUpdateRuntimeParameter = false;
			}

			return runtimeParameter;
		}
		/** ���s���ݒ���\���̂Ŏ擾���� */
		const RuntimeParameter& GetRuntimeParameterByStructure()const
		{
			return runtimeParameter;
		}

		//====================================
		// ���s���ݒ�
		//====================================
	public:
		/** ���s���ݒ���擾����. */
		const SettingData::Standard::IData* GetRuntimeParameter()const
		{
			return pRuntimeParameter;
		}
		SettingData::Standard::IData* GetRuntimeParameter()
		{
			return pRuntimeParameter;
		}

		/** �w�K�ݒ�̃A�C�e�����擾����.
			@param	guid		�擾�Ώۃ��C���[��GUID.	�w�肪�����ꍇ�͑S�Ẵ��C���[�ɑ΂��Ď��s����.
			@param	i_dataID	�ݒ肷��l��ID. */
		SettingData::Standard::IItemBase* GetRuntimeParameterItem(const wchar_t* i_dataID)
		{
			// �w�K�ݒ�f�[�^���擾
			Gravisbell::SettingData::Standard::IData* pLearnSettingData = this->GetRuntimeParameter();
			if(pLearnSettingData == NULL)
				return NULL;

			// �Y��ID�̐ݒ�A�C�e�����擾
			Gravisbell::SettingData::Standard::IItemBase* pItem = pLearnSettingData->GetItemByID(i_dataID);
			if(pItem == NULL)
				return NULL;

			return pItem;
		}

		/** ���s���ݒ��ݒ肷��.
			int�^�Afloat�^�Aenum�^���Ώ�.
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, S32 i_param)
		{
			// �Y��ID�̐ݒ�A�C�e�����擾
			Gravisbell::SettingData::Standard::IItemBase* pItem = this->GetRuntimeParameterItem(i_dataID);
			if(pItem == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

			switch(pItem->GetItemType())
			{
			case Gravisbell::SettingData::Standard::ITEMTYPE_INT:
				{
					Gravisbell::SettingData::Standard::IItem_Int* pItemInt = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Int*>(pItem);
					if(pItemInt == NULL)
						break;
					pItemInt->SetValue(i_param);
					this->onUpdateRuntimeParameter = true;
				}
				return ErrorCode::ERROR_CODE_NONE;

			case Gravisbell::SettingData::Standard::ITEMTYPE_FLOAT:
				{
					Gravisbell::SettingData::Standard::IItem_Float* pItemInt = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Float*>(pItem);
					if(pItemInt == NULL)
						break;
					pItemInt->SetValue((F32)i_param);
					this->onUpdateRuntimeParameter = true;
				}
				return ErrorCode::ERROR_CODE_NONE;

			case Gravisbell::SettingData::Standard::ITEMTYPE_ENUM:
				{
					Gravisbell::SettingData::Standard::IItem_Enum* pItemInt = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Enum*>(pItem);
					if(pItemInt == NULL)
						break;
					pItemInt->SetValue(i_param);
					this->onUpdateRuntimeParameter = true;
				}
				return ErrorCode::ERROR_CODE_NONE;
			}

			return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
		}
		/** ���s���ݒ��ݒ肷��.
			int�^�Afloat�^���Ώ�.
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, F32 i_param)
		{
			// �Y��ID�̐ݒ�A�C�e�����擾
			Gravisbell::SettingData::Standard::IItemBase* pItem = this->GetRuntimeParameterItem(i_dataID);
			if(pItem == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

			switch(pItem->GetItemType())
			{
			case Gravisbell::SettingData::Standard::ITEMTYPE_INT:
				{
					Gravisbell::SettingData::Standard::IItem_Int* pItemInt = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Int*>(pItem);
					if(pItemInt == NULL)
						break;
					pItemInt->SetValue((S32)i_param);
					this->onUpdateRuntimeParameter = true;
				}
				return ErrorCode::ERROR_CODE_NONE;

			case Gravisbell::SettingData::Standard::ITEMTYPE_FLOAT:
				{
					Gravisbell::SettingData::Standard::IItem_Float* pItemInt = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Float*>(pItem);
					if(pItemInt == NULL)
						break;
					pItemInt->SetValue((F32)i_param);
					this->onUpdateRuntimeParameter = true;
				}
				return ErrorCode::ERROR_CODE_NONE;
			}

			return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
		}
		/** ���s���ݒ��ݒ肷��.
			bool�^���Ώ�.
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, bool i_param)
		{
			// �Y��ID�̐ݒ�A�C�e�����擾
			Gravisbell::SettingData::Standard::IItemBase* pItem = this->GetRuntimeParameterItem(i_dataID);
			if(pItem == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

			switch(pItem->GetItemType())
			{
			case Gravisbell::SettingData::Standard::ITEMTYPE_BOOL:
				{
					Gravisbell::SettingData::Standard::IItem_Bool* pItemBool = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Bool*>(pItem);
					if(pItemBool == NULL)
						break;
					pItemBool->SetValue(i_param);
					this->onUpdateRuntimeParameter = true;
				}
				return ErrorCode::ERROR_CODE_NONE;
			}

			return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
		}
		/** ���s���ݒ��ݒ肷��.
			string�^���Ώ�.
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, const wchar_t* i_param)
		{
			// �Y��ID�̐ݒ�A�C�e�����擾
			Gravisbell::SettingData::Standard::IItemBase* pItem = this->GetRuntimeParameterItem(i_dataID);
			if(pItem == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

			switch(pItem->GetItemType())
			{
			case Gravisbell::SettingData::Standard::ITEMTYPE_STRING:
				{
					Gravisbell::SettingData::Standard::IItem_String* pItemString = dynamic_cast<Gravisbell::SettingData::Standard::IItem_String*>(pItem);
					if(pItemString == NULL)
						break;
					pItemString->SetValue(i_param);
					this->onUpdateRuntimeParameter = true;
				}
				return ErrorCode::ERROR_CODE_NONE;
			case Gravisbell::SettingData::Standard::ITEMTYPE_ENUM:
				{
					Gravisbell::SettingData::Standard::IItem_Enum* pItemString = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Enum*>(pItem);
					if(pItemString == NULL)
						break;
					pItemString->SetValue(i_param);
					this->onUpdateRuntimeParameter = true;
				}
				return ErrorCode::ERROR_CODE_NONE;
			}

			return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
		}
	};

	template<typename IOLayer>
	class CLayerBaseRuntimeParameterNone : public CLayerBase<IOLayer>
	{
	protected:
		/** �R���X�g���N�^ */
		CLayerBaseRuntimeParameterNone(Gravisbell::GUID guid)
			:	CLayerBase	(guid)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~CLayerBaseRuntimeParameterNone()
		{
		}

		//====================================
		// ���s���ݒ�
		//====================================
	public:
		/** ���s���ݒ���擾����. */
		const SettingData::Standard::IData* GetRuntimeParameter()const
		{
			return NULL;
		}
		SettingData::Standard::IData* GetRuntimeParameter()
		{
			return NULL;
		}

		/** �w�K�ݒ�̃A�C�e�����擾����.
			@param	guid		�擾�Ώۃ��C���[��GUID.	�w�肪�����ꍇ�͑S�Ẵ��C���[�ɑ΂��Ď��s����.
			@param	i_dataID	�ݒ肷��l��ID. */
		SettingData::Standard::IItemBase* GetRuntimeParameterItem(const wchar_t* i_dataID)
		{
			return NULL;
		}

		/** ���s���ݒ��ݒ肷��.
			int�^�Afloat�^�Aenum�^���Ώ�.
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, S32 i_param)
		{
			return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
		}
		/** ���s���ݒ��ݒ肷��.
			int�^�Afloat�^���Ώ�.
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, F32 i_param)
		{
			return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
		}
		/** ���s���ݒ��ݒ肷��.
			bool�^���Ώ�.
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, bool i_param)
		{
			return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
		}
		/** ���s���ݒ��ݒ肷��.
			string�^���Ώ�.
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, const wchar_t* i_param)
		{
			return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
		}
	};



	template<typename... RuntimeParameter>
	class CNNSingle2SingleLayerBase : public CLayerBaseRuntimeParameter<INNSingle2SingleLayer, RuntimeParameter...>
	{
	private:
		IODataStruct inputDataStruct;	/**< ���̓f�[�^�\�� */
		IODataStruct outputDataStruct;	/**< �o�̓f�[�^�\�� */

	public:
		/** �R���X�g���N�^ */
		CNNSingle2SingleLayerBase(Gravisbell::GUID guid, SettingData::Standard::IData* i_pRuntimeParameter, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
			:	CLayerBaseRuntimeParameter<INNSingle2SingleLayer, RuntimeParameter...>	(guid, i_pRuntimeParameter)
			,	inputDataStruct															(i_inputDataStruct)
			,	outputDataStruct														(i_outputDataStruct)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~CNNSingle2SingleLayerBase()
		{
		}


	public:
		//===========================
		// ���C���[����
		//===========================
		/** ���C���[��ʂ̎擾.
			ELayerKind �̑g�ݍ��킹. */
		U32 GetLayerKindBase(void)const
		{
			return Gravisbell::Layer::ELayerKind::LAYER_KIND_NEURALNETWORK | Gravisbell::Layer::ELayerKind::LAYER_KIND_SINGLE_INPUT | Gravisbell::Layer::ELayerKind::LAYER_KIND_SINGLE_OUTPUT;
		}

	public:
		//===========================
		// ���̓��C���[�֘A
		//===========================
		/** ���̓f�[�^�\�����擾����.
			@return	���̓f�[�^�\�� */
		IODataStruct GetInputDataStruct()const
		{
			return this->inputDataStruct;
		}

		/** ���̓o�b�t�@�����擾����. */
		unsigned int GetInputBufferCount()const
		{
			return this->GetInputDataStruct().GetDataCount();
		}


	public:
		//===========================
		// �o�̓��C���[�֘A
		//===========================
		/** �o�̓f�[�^�\�����擾���� */
		IODataStruct GetOutputDataStruct()const
		{
			return this->outputDataStruct;
		}

		/** �o�̓o�b�t�@�����擾���� */
		unsigned int GetOutputBufferCount()const
		{
			return this->GetOutputDataStruct().GetDataCount();
		}
	};
	template<>
	class CNNSingle2SingleLayerBase<> : public CLayerBaseRuntimeParameterNone<INNSingle2SingleLayer>
	{
	private:
		IODataStruct inputDataStruct;	/**< ���̓f�[�^�\�� */
		IODataStruct outputDataStruct;	/**< �o�̓f�[�^�\�� */

	public:
		/** �R���X�g���N�^ */
		CNNSingle2SingleLayerBase(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
			:	CLayerBaseRuntimeParameterNone<INNSingle2SingleLayer>	(guid)
			,	inputDataStruct											(i_inputDataStruct)
			,	outputDataStruct										(i_outputDataStruct)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~CNNSingle2SingleLayerBase()
		{
		}


	public:
		//===========================
		// ���C���[����
		//===========================
		/** ���C���[��ʂ̎擾.
			ELayerKind �̑g�ݍ��킹. */
		U32 GetLayerKindBase(void)const
		{
			return Gravisbell::Layer::ELayerKind::LAYER_KIND_NEURALNETWORK | Gravisbell::Layer::ELayerKind::LAYER_KIND_SINGLE_INPUT | Gravisbell::Layer::ELayerKind::LAYER_KIND_SINGLE_OUTPUT;
		}

	public:
		//===========================
		// ���̓��C���[�֘A
		//===========================
		/** ���̓f�[�^�\�����擾����.
			@return	���̓f�[�^�\�� */
		IODataStruct GetInputDataStruct()const
		{
			return this->inputDataStruct;
		}

		/** ���̓o�b�t�@�����擾����. */
		unsigned int GetInputBufferCount()const
		{
			return this->GetInputDataStruct().GetDataCount();
		}


	public:
		//===========================
		// �o�̓��C���[�֘A
		//===========================
		/** �o�̓f�[�^�\�����擾���� */
		IODataStruct GetOutputDataStruct()const
		{
			return this->outputDataStruct;
		}

		/** �o�̓o�b�t�@�����擾���� */
		unsigned int GetOutputBufferCount()const
		{
			return this->GetOutputDataStruct().GetDataCount();
		}
	};


	
	template<typename... RuntimeParameter>
	class CNNSingle2MultLayerBase : public CLayerBaseRuntimeParameter<INNSingle2MultLayer, RuntimeParameter...>
	{
	private:
		IODataStruct inputDataStruct;	/**< ���̓f�[�^�\�� */
		IODataStruct outputDataStruct;	/**< �o�̓f�[�^�\�� */

	public:
		/** �R���X�g���N�^ */
		CNNSingle2MultLayerBase(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
			:	CLayerBaseRuntimeParameter<INNSingle2MultLayer, RuntimeParameter...>		(guid)
			,	inputDataStruct																(i_inputDataStruct)
			,	outputDataStruct															(i_outputDataStruct)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~CNNSingle2MultLayerBase()
		{
		}


	public:
		//===========================
		// ���C���[����
		//===========================
		/** ���C���[��ʂ̎擾.
			ELayerKind �̑g�ݍ��킹. */
		U32 GetLayerKindBase(void)const
		{
			return Gravisbell::Layer::ELayerKind::LAYER_KIND_NEURALNETWORK | Gravisbell::Layer::ELayerKind::LAYER_KIND_SINGLE_INPUT | Gravisbell::Layer::ELayerKind::LAYER_KIND_MULT_OUTPUT;
		}

	public:
		//===========================
		// ���̓��C���[�֘A
		//===========================
		/** ���̓f�[�^�\�����擾����.
			@return	���̓f�[�^�\�� */
		IODataStruct GetInputDataStruct()const
		{
			return this->inputDataStruct;
		}

		/** ���̓o�b�t�@�����擾����. */
		unsigned int GetInputBufferCount()const
		{
			return this->GetInputDataStruct().GetDataCount();
		}


	public:
		//===========================
		// �o�̓��C���[�֘A
		//===========================
		/** �o�̓f�[�^�\�����擾���� */
		IODataStruct GetOutputDataStruct()const
		{
			return this->outputDataStruct;
		}

		/** �o�̓o�b�t�@�����擾���� */
		unsigned int GetOutputBufferCount()const
		{
			return this->GetOutputDataStruct().GetDataCount();
		}
	};
	template<>
	class CNNSingle2MultLayerBase<> : public CLayerBaseRuntimeParameterNone<INNSingle2MultLayer>
	{
	private:
		IODataStruct inputDataStruct;	/**< ���̓f�[�^�\�� */
		IODataStruct outputDataStruct;	/**< �o�̓f�[�^�\�� */

	public:
		/** �R���X�g���N�^ */
		CNNSingle2MultLayerBase(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
			:	CLayerBaseRuntimeParameterNone<INNSingle2MultLayer>	(guid)
			,	inputDataStruct										(i_inputDataStruct)
			,	outputDataStruct									(i_outputDataStruct)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~CNNSingle2MultLayerBase()
		{
		}


	public:
		//===========================
		// ���C���[����
		//===========================
		/** ���C���[��ʂ̎擾.
			ELayerKind �̑g�ݍ��킹. */
		U32 GetLayerKindBase(void)const
		{
			return Gravisbell::Layer::ELayerKind::LAYER_KIND_NEURALNETWORK | Gravisbell::Layer::ELayerKind::LAYER_KIND_SINGLE_INPUT | Gravisbell::Layer::ELayerKind::LAYER_KIND_MULT_OUTPUT;
		}

	public:
		//===========================
		// ���̓��C���[�֘A
		//===========================
		/** ���̓f�[�^�\�����擾����.
			@return	���̓f�[�^�\�� */
		IODataStruct GetInputDataStruct()const
		{
			return this->inputDataStruct;
		}

		/** ���̓o�b�t�@�����擾����. */
		unsigned int GetInputBufferCount()const
		{
			return this->GetInputDataStruct().GetDataCount();
		}


	public:
		//===========================
		// �o�̓��C���[�֘A
		//===========================
		/** �o�̓f�[�^�\�����擾���� */
		IODataStruct GetOutputDataStruct()const
		{
			return this->outputDataStruct;
		}

		/** �o�̓o�b�t�@�����擾���� */
		unsigned int GetOutputBufferCount()const
		{
			return this->GetOutputDataStruct().GetDataCount();
		}
	};



	template<typename... RuntimeParameter>
	class CNNMult2SingleLayerBase : public CLayerBaseRuntimeParameter<INNMult2SingleLayer, RuntimeParameter...>
	{
	private:
		std::vector<IODataStruct>	lpInputDataStruct;	/**< ���̓f�[�^�\�� */
		IODataStruct				outputDataStruct;	/**< �o�̓f�[�^�\�� */

	public:
		/** �R���X�g���N�^ */
		CNNMult2SingleLayerBase(Gravisbell::GUID guid, const std::vector<IODataStruct>& i_lpInputDataStruct, const IODataStruct& i_outputDataStruct)
			:	CLayerBaseRuntimeParameter<INNMult2SingleLayer, RuntimeParameter...>	(guid)
			,	lpInputDataStruct														(i_lpInputDataStruct)
			,	outputDataStruct														(i_outputDataStruct)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~CNNMult2SingleLayerBase()
		{
		}


	public:
		//===========================
		// ���C���[����
		//===========================
		/** ���C���[��ʂ̎擾.
			ELayerKind �̑g�ݍ��킹. */
		U32 GetLayerKindBase(void)const
		{
			return Gravisbell::Layer::ELayerKind::LAYER_KIND_NEURALNETWORK | Gravisbell::Layer::ELayerKind::LAYER_KIND_MULT_INPUT | Gravisbell::Layer::ELayerKind::LAYER_KIND_SINGLE_OUTPUT;
		}

	public:
		//===========================
		// ���̓��C���[�֘A
		//===========================
		/** ���̓f�[�^�̐����擾���� */
		U32 GetInputDataCount()const
		{
			return (U32)this->lpInputDataStruct.size();
		}

		/** ���̓f�[�^�\�����擾����.
			@return	���̓f�[�^�\�� */
		IODataStruct GetInputDataStruct(U32 i_dataNum)const
		{
			if(i_dataNum >= this->GetInputDataCount())
				return IODataStruct(0,0,0,0);

			return this->lpInputDataStruct[i_dataNum];
		}

		/** ���̓o�b�t�@�����擾����. */
		U32 GetInputBufferCount(U32 i_dataNum)const
		{
			return this->GetInputDataStruct(i_dataNum).GetDataCount();
		}



	public:
		//===========================
		// �o�̓��C���[�֘A
		//===========================
		/** �o�̓f�[�^�\�����擾���� */
		IODataStruct GetOutputDataStruct()const
		{
			return this->outputDataStruct;
		}

		/** �o�̓o�b�t�@�����擾���� */
		U32 GetOutputBufferCount()const
		{
			return this->GetOutputDataStruct().GetDataCount();
		}
	};
	template<>
	class CNNMult2SingleLayerBase<> : public CLayerBaseRuntimeParameterNone<INNMult2SingleLayer>
	{
	private:
		std::vector<IODataStruct>	lpInputDataStruct;	/**< ���̓f�[�^�\�� */
		IODataStruct				outputDataStruct;	/**< �o�̓f�[�^�\�� */

	public:
		/** �R���X�g���N�^ */
		CNNMult2SingleLayerBase(Gravisbell::GUID guid, const std::vector<IODataStruct>& i_lpInputDataStruct, const IODataStruct& i_outputDataStruct)
			:	CLayerBaseRuntimeParameterNone<INNMult2SingleLayer>	(guid)
			,	lpInputDataStruct									(i_lpInputDataStruct)
			,	outputDataStruct									(i_outputDataStruct)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~CNNMult2SingleLayerBase()
		{
		}


	public:
		//===========================
		// ���C���[����
		//===========================
		/** ���C���[��ʂ̎擾.
			ELayerKind �̑g�ݍ��킹. */
		U32 GetLayerKindBase(void)const
		{
			return Gravisbell::Layer::ELayerKind::LAYER_KIND_NEURALNETWORK | Gravisbell::Layer::ELayerKind::LAYER_KIND_MULT_INPUT | Gravisbell::Layer::ELayerKind::LAYER_KIND_SINGLE_OUTPUT;
		}

	public:
		//===========================
		// ���̓��C���[�֘A
		//===========================
		/** ���̓f�[�^�̐����擾���� */
		U32 GetInputDataCount()const
		{
			return (U32)this->lpInputDataStruct.size();
		}

		/** ���̓f�[�^�\�����擾����.
			@return	���̓f�[�^�\�� */
		IODataStruct GetInputDataStruct(U32 i_dataNum)const
		{
			if(i_dataNum >= this->GetInputDataCount())
				return IODataStruct(0,0,0,0);

			return this->lpInputDataStruct[i_dataNum];
		}

		/** ���̓o�b�t�@�����擾����. */
		U32 GetInputBufferCount(U32 i_dataNum)const
		{
			return this->GetInputDataStruct(i_dataNum).GetDataCount();
		}



	public:
		//===========================
		// �o�̓��C���[�֘A
		//===========================
		/** �o�̓f�[�^�\�����擾���� */
		IODataStruct GetOutputDataStruct()const
		{
			return this->outputDataStruct;
		}

		/** �o�̓o�b�t�@�����擾���� */
		U32 GetOutputBufferCount()const
		{
			return this->GetOutputDataStruct().GetDataCount();
		}
	};






}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
