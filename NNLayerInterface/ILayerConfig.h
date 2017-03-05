//===========================
// NN�̃��C���[�ݒ荀�ڃt�H�[�}�b�g�x�[�X
//===========================
#ifndef __GRAVISBELL_I_NN_LAYER_CONFIG_H__
#define __GRAVISBELL_I_NN_LAYER_CONFIG_H__

#include<guiddef.h>

#include"Common/Common.h"
#include"Common/ErrorCode.h"


namespace Gravisbell {
namespace NeuralNetwork {

	/** �ݒ荀�ڎ�� */
	enum LayerConfigItemType
	{
		CONFIGITEM_TYPE_FLOAT,
		CONFIGITEM_TYPE_INT,
		CONFIGITEM_TYPE_STRING,
		CONFIGITEM_TYPE_BOOL,
		CONFIGITEM_TYPE_ENUM,

		CONFIGITEM_TYPE_COUNT
	};

	static const U32 CONFIGITEM_NAME_MAX = 64;
	static const U32 CONFIGITEM_TEXT_MAX = 1024;
	static const U32 CONFIGITEM_ID_MAX   = 64;
	static const U32 LAYERTYPE_CODE_MAX  = 36 + 1;	/**< ���C���[�̎�ނ����ʂ��邽�߂̃R�[�h�̍ő啶����(36�����{null�R�[�h) */

	/** �ݒ荀�ڂ̃x�[�X */
	class ILayerConfigItemBase
	{
	public:
		/** �R���X�g���N�^ */
		ILayerConfigItemBase(){};
		/** �f�X�g���N�^ */
		virtual ~ILayerConfigItemBase(){};

		/** ��v���Z */
		virtual bool operator==(const ILayerConfigItemBase& item)const = 0;
		/** �s��v���Z */
		virtual bool operator!=(const ILayerConfigItemBase& item)const = 0;

		/** ���g�̕������쐬���� */
		virtual ILayerConfigItemBase* Clone()const = 0;

	public:
		/** �ݒ荀�ڎ�ʂ��擾���� */
		virtual LayerConfigItemType GetItemType()const = 0;
		/** ����ID���擾����.
			ID��LayerConfig���ɂ����ĕK�����j�[�N�ł���.
			@param o_szIDBuf	ID���i�[����o�b�t�@.CONFIGITEM_ID_MAX�̃o�C�g�����K�v.�@*/
		virtual Gravisbell::ErrorCode GetConfigID(wchar_t o_szIDBuf[])const = 0;
		/** ���ږ����擾����.
			@param o_szNameBuf	���O���i�[����o�b�t�@.CONFIGITEM_NAME_MAX�̃o�C�g�����K�v. */
		virtual Gravisbell::ErrorCode GetConfigName(wchar_t o_szNameBuf[])const = 0;
		/** ���ڂ̐����e�L�X�g���擾����.
			@param o_szBuf	���������i�[����o�b�t�@.CONFIGITEM_TEXT_MAX�̃o�C�g�����K�v. */
		virtual Gravisbell::ErrorCode GetConfigText(wchar_t o_szBuf[])const = 0;

	public:
		//================================
		// �t�@�C���ۑ��֘A.
		// ������{�̂�񋓒l��ID�ȂǍ\���̂ɂ͕ۑ�����Ȃ��ׂ���������舵��.
		//================================

		/** �ۑ��ɕK�v�ȃo�C�g�����擾���� */
		virtual U32 GetUseBufferByteCount()const = 0;

		/** �o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ0 */
		virtual S32 WriteToBuffer(BYTE* o_lpBuffer)const = 0;
		/** �o�b�t�@����ǂݍ���.
			@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
			@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
			@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
		virtual S32 ReadFromBuffer(const BYTE* i_lpBuffer, S32 i_bufferSize) = 0;

	public:
		//================================
		// �\���̂𗘗p�����f�[�^�̎�舵��.
		// ���ʂ����Ȃ�����ɃA�N�Z�X���x������
		//================================

		/** �\���̂ɏ�������.
			@return	�g�p�����o�C�g��. */
		virtual S32 WriteToStruct(BYTE* o_lpBuffer)const = 0;
		/** �\���̂���ǂݍ���.
			@return	�g�p�����o�C�g��. */
		virtual S32 ReadFromStruct(const BYTE* i_lpBuffer) = 0;
	};

	/** �ݒ荀��(����) */
	class ILayerConfigItem_Float : virtual public ILayerConfigItemBase
	{
	public:
		/** �R���X�g���N�^ */
		ILayerConfigItem_Float()
			: ILayerConfigItemBase()
		{};
		/** �f�X�g���N�^ */
		virtual ~ILayerConfigItem_Float(){};

	public:
		/** �l���擾���� */
		virtual F32 GetValue()const = 0;
		/** �l��ݒ肷��
			@param value	�ݒ肷��l
			@return ���������ꍇ0 */
		virtual Gravisbell::ErrorCode SetValue(F32 value) = 0;

	public:
		/** �ݒ�\�ŏ��l���擾���� */
		virtual F32 GetMin()const = 0;
		/** �ݒ�\�ő�l���擾���� */
		virtual F32 GetMax()const = 0;

		/** �f�t�H���g�̐ݒ�l���擾���� */
		virtual F32 GetDefault()const = 0;
	};

	/** �ݒ荀��(����) */
	class ILayerConfigItem_Int : virtual public ILayerConfigItemBase
	{
	public:
		/** �R���X�g���N�^ */
		ILayerConfigItem_Int()
			: ILayerConfigItemBase()
		{};
		/** �f�X�g���N�^ */
		virtual ~ILayerConfigItem_Int(){};

	public:
		/** �l���擾���� */
		virtual S32 GetValue()const = 0;
		/** �l��ݒ肷��
			@param value	�ݒ肷��l
			@return ���������ꍇ0 */
		virtual Gravisbell::ErrorCode SetValue(S32 value) = 0;
		
	public:
		/** �ݒ�\�ŏ��l���擾���� */
		virtual S32 GetMin()const = 0;
		/** �ݒ�\�ő�l���擾���� */
		virtual S32 GetMax()const = 0;

		/** �f�t�H���g�̐ݒ�l���擾���� */
		virtual S32 GetDefault()const = 0;
	};

	/** �ݒ荀��(������) */
	class ILayerConfigItem_String : virtual public ILayerConfigItemBase
	{
	public:
		/** �R���X�g���N�^ */
		ILayerConfigItem_String()
			: ILayerConfigItemBase()
		{};
		/** �f�X�g���N�^ */
		virtual ~ILayerConfigItem_String(){};

	public:
		/** ������̒������擾���� */
		virtual U32 GetLength()const = 0;
		/** �l���擾����.
			@param o_szBuf	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		virtual S32 GetValue(wchar_t o_szBuf[])const = 0;
		/** �l���擾����.
			@return ���������ꍇ������̐擪�A�h���X. */
		virtual const wchar_t* GetValue()const = 0;
		/** �l��ݒ肷��
			@param i_szBuf	�ݒ肷��l
			@return ���������ꍇ0 */
		virtual Gravisbell::ErrorCode SetValue(const wchar_t i_szBuf[]) = 0;
		
	public:
		/** �f�t�H���g�̐ݒ�l���擾����
			@param o_szBuf	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		virtual S32 GetDefault(wchar_t o_szBuf[])const = 0;
	};

	/** �ݒ荀��(�_���l) */
	class ILayerConfigItem_Bool : virtual public ILayerConfigItemBase
	{
	public:
		/** �R���X�g���N�^ */
		ILayerConfigItem_Bool()
			: ILayerConfigItemBase()
		{};
		/** �f�X�g���N�^ */
		virtual ~ILayerConfigItem_Bool(){};

	public:
		/** �l���擾���� */
		virtual bool GetValue()const = 0;
		/** �l��ݒ肷��
			@param value	�ݒ肷��l
			@return ���������ꍇ0 */
		virtual Gravisbell::ErrorCode SetValue(bool value) = 0;

	public:
		/** �f�t�H���g�̐ݒ�l���擾���� */
		virtual bool GetDefault()const = 0;
	};

	/** �ݒ荀��(�񋓌^).
		���񋓗v�fID=��ӂ̉p����. �񋓗v�f��=���p�҂��F�����₷�����O.������.(�d����).1�s. �R�����g=�񋓗v�f�̐�����.�����s�� */
	class ILayerConfigItem_Enum : virtual public ILayerConfigItemBase
	{
	public:
		static const U32 ID_BUFFER_MAX = 64;			/**< �񋓗v�fID�̍ő啶���� */
		static const U32 NAME_BUFFER_MAX = 64;			/**< �񋓗v�f���̍ő啶���� */
		static const U32 COMMENT_BUFFER_MAX = 256;		/**< �R�����g�̍ő啶���� */

	public:
		/** �R���X�g���N�^ */
		ILayerConfigItem_Enum()
			: ILayerConfigItemBase()
		{};
		/** �f�X�g���N�^ */
		virtual ~ILayerConfigItem_Enum(){};

	public:
		/** �l���擾���� */
		virtual S32 GetValue()const = 0;

		/** �l��ݒ肷��
			@param value	�ݒ肷��l */
		virtual Gravisbell::ErrorCode SetValue(S32 value) = 0;
		/** �l��ݒ肷��(������w��)
			@param i_szID	�ݒ肷��l(������w��) */
		virtual Gravisbell::ErrorCode SetValue(const wchar_t i_szEnumID[]) = 0;

	public:
		/** �񋓗v�f�����擾���� */
		virtual U32 GetEnumCount()const = 0;
		/** �񋓗v�fID��ԍ��w��Ŏ擾����.
			@param num		�v�f�ԍ�
			@param o_szBuf	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		virtual S32 GetEnumID(S32 num, wchar_t o_szBuf[])const = 0;
		/** �񋓗v�fID��ԍ��w��Ŏ擾����.
			@param num		�v�f�ԍ�
			@param o_szBuf	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		virtual S32 GetEnumName(S32 num, wchar_t o_szBuf[])const = 0;
		/** �񋓗v�f��������ԍ��w��Ŏ擾����.
			@param num		�v�f�ԍ�
			@param o_szBuf	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		virtual S32 GetEnumText(S32 num, wchar_t o_szBuf[])const = 0;

		/** ID���w�肵�ė񋓗v�f�ԍ����擾����
			@param i_szBuf�@���ׂ��ID.
			@return ���������ꍇ0<=num<GetEnumCount�̒l. ���s�����ꍇ�͕��̒l���Ԃ�. */
		virtual S32 GetNumByID(const wchar_t i_szBuf[])const = 0;


		/** �f�t�H���g�̐ݒ�l���擾���� */
		virtual S32 GetDefault()const = 0;
	};



	/** �ݒ��� */
	class ILayerConfig
	{
	public:
		/** �R���X�g���N�^ */
		ILayerConfig(){}
		/** �f�X�g���N�^ */
		virtual ~ILayerConfig(){}

		/** ��v���Z */
		virtual bool operator==(const ILayerConfig& config)const = 0;
		/** �s��v���Z */
		virtual bool operator!=(const ILayerConfig& config)const = 0;

		/** ���g�̕������쐬���� */
		virtual ILayerConfig* Clone()const = 0;

	public:
		/** ���C���[���ʃR�[�h���擾����.
			@param o_guid	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		virtual Gravisbell::ErrorCode GetLayerCode(GUID& o_guid)const = 0;

	public:
		/** �ݒ荀�ڐ����擾���� */
		virtual U32 GetItemCount()const = 0;
		/** �ݒ荀�ڂ�ԍ��w��Ŏ擾���� */
		virtual ILayerConfigItemBase* GetItemByNum(U32 i_num) = 0;
		/** �ݒ荀�ڂ�ԍ��w��Ŏ擾���� */
		virtual const ILayerConfigItemBase* GetItemByNum(U32 i_num)const = 0;
		/** �ݒ荀�ڂ�ID�w��Ŏ擾���� */
		virtual const ILayerConfigItemBase* GetItemByID(const wchar_t i_szIDBuf[])const = 0;

	public:
		/** �ۑ��ɕK�v�ȃo�C�g�����擾���� */
		virtual U32 GetUseBufferByteCount()const = 0;

		/** �o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ0 */
		virtual S32 WriteToBuffer(BYTE* o_lpBuffer)const = 0;

	public:
		/** �\���̂Ƀf�[�^���i�[����.
			@param	o_lpBuffer	�\���̂̐擪�A�h���X. �\���̂�ConvertNNCofigToSource����o�͂��ꂽ�\�[�X���g�p����. */
		virtual ErrorCode WriteToStruct(BYTE* o_lpBuffer)const = 0;
		/** �\���̂���f�[�^��ǂݍ���.
			@param	i_lpBuffer	�\���̂̐擪�A�h���X. �\���̂�ConvertNNCofigToSource����o�͂��ꂽ�\�[�X���g�p����. */
		virtual ErrorCode ReadFromStruct(const BYTE* i_lpBuffer) = 0;
	};


}	// NeuralNetwork
}	// Gravisbell



#endif	// __I_NN_LAYER_CONFIG_H__