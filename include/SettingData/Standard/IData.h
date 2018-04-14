//===========================
// NN�̃��C���[�ݒ荀�ڃt�H�[�}�b�g�x�[�X
//===========================
#ifndef __GRAVISBELL_I_NN_LAYER_CONFIG_H__
#define __GRAVISBELL_I_NN_LAYER_CONFIG_H__

#include<guiddef.h>

#include"../../Common/Common.h"
#include"../../Common/ErrorCode.h"
#include"../../Common/Guiddef.h"


namespace Gravisbell {
namespace SettingData {
namespace Standard {

	/** �ݒ荀�ڎ�� */
	enum ItemType
	{
		ITEMTYPE_FLOAT,
		ITEMTYPE_INT,
		ITEMTYPE_STRING,
		ITEMTYPE_BOOL,
		ITEMTYPE_ENUM,
		ITEMTYPE_VECTOR2D_FLOAT,
		ITEMTYPE_VECTOR2D_INT,
		ITEMTYPE_VECTOR3D_FLOAT,
		ITEMTYPE_VECTOR3D_INT,

		ITEMTYPE_COUNT
	};

	static const U32 ITEM_NAME_MAX = 64;
	static const U32 ITEM_TEXT_MAX = 1024;
	static const U32 ITEM_ID_MAX   = 64;
	static const U32 LAYERTYPE_CODE_MAX  = 36 + 1;	/**< ���C���[�̎�ނ����ʂ��邽�߂̃R�[�h�̍ő啶����(36�����{null�R�[�h) */

	/** �ݒ荀�ڂ̃x�[�X */
	class IItemBase
	{
	public:
		/** �R���X�g���N�^ */
		IItemBase(){};
		/** �f�X�g���N�^ */
		virtual ~IItemBase(){};

		/** ��v���Z */
		virtual bool operator==(const IItemBase& item)const = 0;
		/** �s��v���Z */
		virtual bool operator!=(const IItemBase& item)const = 0;

		/** ���g�̕������쐬���� */
		virtual IItemBase* Clone()const = 0;

	public:
		/** �ݒ荀�ڎ�ʂ��擾���� */
		virtual ItemType GetItemType()const = 0;
		/** ����ID���擾����.
			ID��LayerConfig���ɂ����ĕK�����j�[�N�ł���.
			@param o_szIDBuf	ID���i�[����o�b�t�@.ITEM_ID_MAX�̃o�C�g�����K�v.�@*/
		virtual Gravisbell::ErrorCode GetConfigID(wchar_t o_szIDBuf[])const = 0;
		/** ���ږ����擾����.
			@param o_szNameBuf	���O���i�[����o�b�t�@.ITEM_NAME_MAX�̃o�C�g�����K�v. */
		virtual Gravisbell::ErrorCode GetConfigName(wchar_t o_szNameBuf[])const = 0;
		/** ���ڂ̐����e�L�X�g���擾����.
			@param o_szBuf	���������i�[����o�b�t�@.ITEM_TEXT_MAX�̃o�C�g�����K�v. */
		virtual Gravisbell::ErrorCode GetConfigText(wchar_t o_szBuf[])const = 0;

	public:
		//================================
		// �t�@�C���ۑ��֘A.
		// ������{�̂�񋓒l��ID�ȂǍ\���̂ɂ͕ۑ�����Ȃ��ׂ���������舵��.
		//================================

		/** �ۑ��ɕK�v�ȃo�C�g�����擾���� */
		virtual U64 GetUseBufferByteCount()const = 0;

		/** �o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
		virtual S64 WriteToBuffer(BYTE* o_lpBuffer)const = 0;
		/** �o�b�t�@����ǂݍ���.
			@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
			@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
			@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
		virtual S64 ReadFromBuffer(const BYTE* i_lpBuffer, S64 i_bufferSize) = 0;

	public:
		//================================
		// �\���̂𗘗p�����f�[�^�̎�舵��.
		// ���ʂ����Ȃ�����ɃA�N�Z�X���x������
		//================================
		/** �A���C�����g�̃T�C�Y���擾���� */
		virtual U64 GetAlignmentByteCount()const = 0;

		/** �\���̂ɏ�������.
			@return	�g�p�����o�C�g��. */
		virtual S32 WriteToStruct(BYTE* o_lpBuffer)const = 0;
		/** �\���̂���ǂݍ���.
			@return	�g�p�����o�C�g��. */
		virtual S32 ReadFromStruct(const BYTE* i_lpBuffer) = 0;
	};

	/** �ݒ荀��(����) */
	class IItem_Float : virtual public IItemBase
	{
	public:
		/** �R���X�g���N�^ */
		IItem_Float()
			: IItemBase()
		{};
		/** �f�X�g���N�^ */
		virtual ~IItem_Float(){};

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
	class IItem_Int : virtual public IItemBase
	{
	public:
		/** �R���X�g���N�^ */
		IItem_Int()
			: IItemBase()
		{};
		/** �f�X�g���N�^ */
		virtual ~IItem_Int(){};

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
	class IItem_String : virtual public IItemBase
	{
	public:
		/** �R���X�g���N�^ */
		IItem_String()
			: IItemBase()
		{};
		/** �f�X�g���N�^ */
		virtual ~IItem_String(){};

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
	class IItem_Bool : virtual public IItemBase
	{
	public:
		/** �R���X�g���N�^ */
		IItem_Bool()
			: IItemBase()
		{};
		/** �f�X�g���N�^ */
		virtual ~IItem_Bool(){};

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
	class IItem_Enum : virtual public IItemBase
	{
	public:
		static const U32 ID_BUFFER_MAX = 64;			/**< �񋓗v�fID�̍ő啶���� */
		static const U32 NAME_BUFFER_MAX = 64;			/**< �񋓗v�f���̍ő啶���� */
		static const U32 COMMENT_BUFFER_MAX = 256;		/**< �R�����g�̍ő啶���� */

	public:
		/** �R���X�g���N�^ */
		IItem_Enum()
			: IItemBase()
		{};
		/** �f�X�g���N�^ */
		virtual ~IItem_Enum(){};

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

	/** �ݒ荀��(Vector3)(����) */
	template<class Type>
	class IItem_Vector3D : public IItemBase
	{
	public:
		/** �R���X�g���N�^ */
		IItem_Vector3D()
			: IItemBase()
		{};
		/** �f�X�g���N�^ */
		virtual ~IItem_Vector3D(){};

	public:
		/** �l���擾���� */
		virtual const Vector3D<Type>& GetValue()const = 0;
		virtual Type GetValueX()const = 0;
		virtual Type GetValueY()const = 0;
		virtual Type GetValueZ()const = 0;
		/** �l��ݒ肷��
			@param value	�ݒ肷��l
			@return ���������ꍇ0 */
		virtual Gravisbell::ErrorCode SetValue(const Vector3D<Type>& value) = 0;
		virtual Gravisbell::ErrorCode SetValueX(Type value) = 0;
		virtual Gravisbell::ErrorCode SetValueY(Type value) = 0;
		virtual Gravisbell::ErrorCode SetValueZ(Type value) = 0;

	public:
		/** �ݒ�\�ŏ��l���擾���� */
		virtual Type GetMinX()const = 0;
		virtual Type GetMinY()const = 0;
		virtual Type GetMinZ()const = 0;
		/** �ݒ�\�ő�l���擾���� */
		virtual Type GetMaxX()const = 0;
		virtual Type GetMaxY()const = 0;
		virtual Type GetMaxZ()const = 0;

		/** �f�t�H���g�̐ݒ�l���擾���� */
		virtual Type GetDefaultX()const = 0;
		virtual Type GetDefaultY()const = 0;
		virtual Type GetDefaultZ()const = 0;
	};
	typedef IItem_Vector3D<F32> IItem_Vector3D_Float;
	typedef IItem_Vector3D<S32> IItem_Vector3D_Int;


	/** �ݒ��� */
	class IData
	{
	public:
		/** �R���X�g���N�^ */
		IData(){}
		/** �f�X�g���N�^ */
		virtual ~IData(){}

		/** ��v���Z */
		virtual bool operator==(const IData& config)const = 0;
		/** �s��v���Z */
		virtual bool operator!=(const IData& config)const = 0;

		/** ���g�̕������쐬���� */
		virtual IData* Clone()const = 0;

	public:
		/** ���C���[���ʃR�[�h���擾����.
			@param o_guid	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		virtual Gravisbell::ErrorCode GetLayerCode(Gravisbell::GUID& o_guid)const = 0;

	public:
		/** �ݒ荀�ڐ����擾���� */
		virtual U32 GetItemCount()const = 0;
		/** �ݒ荀�ڂ�ԍ��w��Ŏ擾���� */
		virtual IItemBase* GetItemByNum(U32 i_num) = 0;
		/** �ݒ荀�ڂ�ԍ��w��Ŏ擾���� */
		virtual const IItemBase* GetItemByNum(U32 i_num)const = 0;
		/** �ݒ荀�ڂ�ID�w��Ŏ擾���� */
		virtual IItemBase* GetItemByID(const wchar_t i_szIDBuf[]) = 0;
		/** �ݒ荀�ڂ�ID�w��Ŏ擾���� */
		virtual const IItemBase* GetItemByID(const wchar_t i_szIDBuf[])const = 0;

	public:
		/** �ۑ��ɕK�v�ȃo�C�g�����擾���� */
		virtual U64 GetUseBufferByteCount()const = 0;

		/** �o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
		virtual S64 WriteToBuffer(BYTE* o_lpBuffer)const = 0;

	public:
		/** �\���̂Ƀf�[�^���i�[����.
			@param	o_lpBuffer	�\���̂̐擪�A�h���X. �\���̂�ConvertNNCofigToSource����o�͂��ꂽ�\�[�X���g�p����. */
		virtual ErrorCode WriteToStruct(BYTE* o_lpBuffer)const = 0;
		/** �\���̂���f�[�^��ǂݍ���.
			@param	i_lpBuffer	�\���̂̐擪�A�h���X. �\���̂�ConvertNNCofigToSource����o�͂��ꂽ�\�[�X���g�p����. */
		virtual ErrorCode ReadFromStruct(const BYTE* i_lpBuffer) = 0;
	};


}	// Standard
}	// SettingData
}	// Gravisbell



#endif	// __I_NN_LAYER_CONFIG_H__