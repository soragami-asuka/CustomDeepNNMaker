//==================================
// �ݒ荀��(Vector3)(����)
//==================================
#include "stdafx.h"

#include"SettingData.h"
#include"ItemBase.h"

#include<string>
#include<vector>

namespace Gravisbell {
namespace SettingData {
namespace Standard {

	template<class Type, ItemType itemType>
	class Item_Vector3D : public ItemBase<IItem_Vector3D<Type>>
	{
	private:
		Vector3D<Type> minValue;
		Vector3D<Type> maxValue;
		Vector3D<Type> defaultValue;

		Vector3D<Type> value;

	public:
		/** �R���X�g���N�^ */
		Item_Vector3D(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], Vector3D<Type> minValue, Vector3D<Type> maxValue, Vector3D<Type> defaultValue)
			: ItemBase(i_szID, i_szName, i_szText)
			, minValue(minValue)
			, maxValue(maxValue)
			, defaultValue(defaultValue)
			, value	(defaultValue)
		{
		}
		/** �R�s�[�R���X�g���N�^ */
		Item_Vector3D(const Item_Vector3D& item)
			: ItemBase(item)
			, minValue(item.minValue)
			, maxValue(item.maxValue)
			, defaultValue(item.defaultValue)
			, value	(item.value)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~Item_Vector3D(){}
		
		/** ��v���Z */
		bool operator==(const IItemBase& item)const
		{
			// ��ʂ̊m�F
			if(this->GetItemType() != item.GetItemType())
				return false;

			// �A�C�e����ϊ�
			const Item_Vector3D* pItem = dynamic_cast<const Item_Vector3D*>(&item);
			if(pItem == NULL)
				return false;

			// �x�[�X��r
			if(ItemBase::operator!=(*pItem))
				return false;

			if(this->minValue != pItem->minValue)
				return false;
			if(this->maxValue != pItem->maxValue)
				return false;
			if(this->defaultValue != pItem->defaultValue)
				return false;

			if(this->value != pItem->value)
				return false;

			return true;
		}
		/** �s��v���Z */
		bool operator!=(const IItemBase& item)const
		{
			return !(*this == item);
		}

		/** ���g�̕������쐬���� */
		IItemBase* Clone()const
		{
			return new Item_Vector3D(*this);
		}

	public:
		/** �ݒ荀�ڎ�ʂ��擾���� */
		ItemType GetItemType()const
		{
			return ITEMTYPE_FLOAT;
		}

	public:
		/** �l���擾���� */
		const Vector3D<Type>& GetValue()const
		{
			return this->value;
		}
		Type GetValueX()const
		{
			return this->value.x;
		}
		Type GetValueY()const
		{
			return this->value.y;
		}
		Type GetValueZ()const
		{
			return this->value.z;
		}
		/** �l��ݒ肷��
			@param value	�ݒ肷��l
			@return ���������ꍇ0 */
		Gravisbell::ErrorCode SetValue(const Vector3D<Type>& value)
		{
			// X��
			if(value.x < this->minValue.x)
				return ERROR_CODE_COMMON_OUT_OF_VALUERANGE;
			if(value.x > this->maxValue.x)
				return ERROR_CODE_COMMON_OUT_OF_VALUERANGE;
			
			// Y��
			if(value.y < this->minValue.y)
				return ERROR_CODE_COMMON_OUT_OF_VALUERANGE;
			if(value.y > this->maxValue.y)
				return ERROR_CODE_COMMON_OUT_OF_VALUERANGE;
			
			// Z��
			if(value.z < this->minValue.z)
				return ERROR_CODE_COMMON_OUT_OF_VALUERANGE;
			if(value.z > this->maxValue.z)
				return ERROR_CODE_COMMON_OUT_OF_VALUERANGE;

			this->value = value;

			return ErrorCode::ERROR_CODE_NONE;
		}
		ErrorCode SetValueX(Type value)
		{
			if(value < this->minValue.x)
				return ERROR_CODE_COMMON_OUT_OF_VALUERANGE;
			if(value > this->maxValue.x)
				return ERROR_CODE_COMMON_OUT_OF_VALUERANGE;

			this->value.x = value;

			return ERROR_CODE_NONE;
		}
		ErrorCode SetValueY(Type value)
		{
			if(value < this->minValue.y)
				return ERROR_CODE_COMMON_OUT_OF_VALUERANGE;
			if(value > this->maxValue.y)
				return ERROR_CODE_COMMON_OUT_OF_VALUERANGE;

			this->value.y = value;

			return ERROR_CODE_NONE;
		}
		ErrorCode SetValueZ(Type value)
		{
			if(value < this->minValue.z)
				return ERROR_CODE_COMMON_OUT_OF_VALUERANGE;
			if(value > this->maxValue.z)
				return ERROR_CODE_COMMON_OUT_OF_VALUERANGE;

			this->value.z = value;

			return ERROR_CODE_NONE;
		}

	public:
		/** �ݒ�\�ŏ��l���擾���� */
		Type GetMinX()const
		{
			return this->minValue.x;
		}
		Type GetMinY()const
		{
			return this->minValue.y;
		}
		Type GetMinZ()const
		{
			return this->minValue.z;
		}
		/** �ݒ�\�ő�l���擾���� */
		Type GetMaxX()const
		{
			return this->maxValue.x;
		}
		Type GetMaxY()const
		{
			return this->maxValue.y;
		}
		Type GetMaxZ()const
		{
			return this->maxValue.z;
		}

		/** �f�t�H���g�̐ݒ�l���擾���� */
		Type GetDefaultX()const
		{
			return this->defaultValue.x;
		}
		Type GetDefaultY()const
		{
			return this->defaultValue.y;
		}
		Type GetDefaultZ()const
		{
			return this->defaultValue.z;
		}
		

	public:
		//================================
		// �t�@�C���ۑ��֘A.
		// ������{�̂�񋓒l��ID�ȂǍ\���̂ɂ͕ۑ�����Ȃ��ׂ���������舵��.
		//================================

		/** �ۑ��ɕK�v�ȃo�C�g�����擾���� */
		U32 GetUseBufferByteCount()const
		{
			U32 byteCount = 0;

			byteCount += sizeof(this->value);			// �l

			return byteCount;
		}

		/** �o�b�t�@����ǂݍ���.
			@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
			@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
			@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
		S32 ReadFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize)
		{
			if(i_bufferSize < (int)this->GetUseBufferByteCount())
				return -1;

			U32 bufferPos = 0;

			// �l
			Vector3D<Type> value = *(Vector3D<Type>*)&i_lpBuffer[bufferPos];
			bufferPos += sizeof(this->value);

			this->SetValueX(value.x);
			this->SetValueY(value.y);
			this->SetValueZ(value.z);

			return bufferPos;
		}
		/** �o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
		S32 WriteToBuffer(BYTE* o_lpBuffer)const
		{
			U32 bufferPos = 0;

			// �l
			*(Vector3D<Type>*)&o_lpBuffer[bufferPos] = this->value;
			bufferPos += sizeof(this->value);

			return bufferPos;
		}

	public:
		//================================
		// �\���̂𗘗p�����f�[�^�̎�舵��.
		// ���ʂ����Ȃ�����ɃA�N�Z�X���x������
		//================================

		/** �\���̂ɏ�������.
			@return	�g�p�����o�C�g��. */
		S32 WriteToStruct(BYTE* o_lpBuffer)const
		{
			Vector3D<Type> value = this->GetValue();

			*(Vector3D<Type>*)o_lpBuffer = value;

			return sizeof(Vector3D<Type>);
		}
		/** �\���̂���ǂݍ���.
			@return	�g�p�����o�C�g��. */
		S32 ReadFromStruct(const BYTE* i_lpBuffer)
		{
			Vector3D<Type> value = *(const Vector3D<Type>*)i_lpBuffer;

			this->SetValue(value);

			return sizeof(Vector3D<Type>);
		}
	};
	typedef Item_Vector3D<F32, ITEMTYPE_VECTOR3D_FLOAT> Item_Vector3D_Float;
	typedef Item_Vector3D<S32, ITEMTYPE_VECTOR3D_INT> Item_Vector3D_Int;

	/** �ݒ荀��(Vector3)(����)���쐬����. */
	extern "C" GRAVISBELL_SETTINGDATA_API IItem_Vector3D_Float* CreateItem_Vector3D_Float(
		const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[],
		float i_minValueX, float i_minValueY, float i_minValueZ,
		float i_maxValueX, float i_maxValueY, float i_maxValueZ,
		float i_defaultValueX, float i_defaultValueY, float i_defaultValueZ)
	{
		return new Item_Vector3D_Float(
			i_szID, i_szName, i_szText,
			Vector3D<F32>(i_minValueX, i_minValueY, i_minValueZ),
			Vector3D<F32>(i_maxValueX, i_maxValueY, i_maxValueZ),
			Vector3D<F32>(i_defaultValueX, i_defaultValueY, i_defaultValueZ));
	}
	
	/** �ݒ荀��(Vector3)(����)���쐬����. */
	extern "C" GRAVISBELL_SETTINGDATA_API IItem_Vector3D_Int* CreateItem_Vector3D_Int(
		const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[],
		S32 i_minValueX, S32 i_minValueY, S32 i_minValueZ,
		S32 i_maxValueX, S32 i_maxValueY, S32 i_maxValueZ,
		S32 i_defaultValueX, S32 i_defaultValueY, S32 i_defaultValueZ)
	{
		return new Item_Vector3D_Int(
			i_szID, i_szName, i_szText,
			Vector3D<S32>(i_minValueX, i_minValueY, i_minValueZ),
			Vector3D<S32>(i_maxValueX, i_maxValueY, i_maxValueZ),
			Vector3D<S32>(i_defaultValueX, i_defaultValueY, i_defaultValueZ));
	}

}	// Standard
}	// SettingData
}	// Gravisbell