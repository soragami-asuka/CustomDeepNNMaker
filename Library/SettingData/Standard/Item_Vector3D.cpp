//==================================
// 設定項目(Vector3)(実数)
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
		/** コンストラクタ */
		Item_Vector3D(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], Vector3D<Type> minValue, Vector3D<Type> maxValue, Vector3D<Type> defaultValue)
			: ItemBase(i_szID, i_szName, i_szText)
			, minValue(minValue)
			, maxValue(maxValue)
			, defaultValue(defaultValue)
			, value	(defaultValue)
		{
		}
		/** コピーコンストラクタ */
		Item_Vector3D(const Item_Vector3D& item)
			: ItemBase(item)
			, minValue(item.minValue)
			, maxValue(item.maxValue)
			, defaultValue(item.defaultValue)
			, value	(item.value)
		{
		}
		/** デストラクタ */
		virtual ~Item_Vector3D(){}
		
		/** 一致演算 */
		bool operator==(const IItemBase& item)const
		{
			// 種別の確認
			if(this->GetItemType() != item.GetItemType())
				return false;

			// アイテムを変換
			const Item_Vector3D* pItem = dynamic_cast<const Item_Vector3D*>(&item);
			if(pItem == NULL)
				return false;

			// ベース比較
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
		/** 不一致演算 */
		bool operator!=(const IItemBase& item)const
		{
			return !(*this == item);
		}

		/** 自身の複製を作成する */
		IItemBase* Clone()const
		{
			return new Item_Vector3D(*this);
		}

	public:
		/** 設定項目種別を取得する */
		ItemType GetItemType()const
		{
			return ITEMTYPE_FLOAT;
		}

	public:
		/** 値を取得する */
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
		/** 値を設定する
			@param value	設定する値
			@return 成功した場合0 */
		Gravisbell::ErrorCode SetValue(const Vector3D<Type>& value)
		{
			// X軸
			if(value.x < this->minValue.x)
				return ERROR_CODE_COMMON_OUT_OF_VALUERANGE;
			if(value.x > this->maxValue.x)
				return ERROR_CODE_COMMON_OUT_OF_VALUERANGE;
			
			// Y軸
			if(value.y < this->minValue.y)
				return ERROR_CODE_COMMON_OUT_OF_VALUERANGE;
			if(value.y > this->maxValue.y)
				return ERROR_CODE_COMMON_OUT_OF_VALUERANGE;
			
			// Z軸
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
		/** 設定可能最小値を取得する */
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
		/** 設定可能最大値を取得する */
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

		/** デフォルトの設定値を取得する */
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
		// ファイル保存関連.
		// 文字列本体や列挙値のIDなど構造体には保存されない細かい情報を取り扱う.
		//================================

		/** 保存に必要なバイト数を取得する */
		U32 GetUseBufferByteCount()const
		{
			U32 byteCount = 0;

			byteCount += sizeof(this->value);			// 値

			return byteCount;
		}

		/** バッファから読み込む.
			@param i_lpBuffer	読み込みバッファの先頭アドレス.
			@param i_bufferSize	読み込み可能バッファのサイズ.
			@return	実際に読み取ったバッファサイズ. 失敗した場合は負の値 */
		S32 ReadFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize)
		{
			if(i_bufferSize < (int)this->GetUseBufferByteCount())
				return -1;

			U32 bufferPos = 0;

			// 値
			Vector3D<Type> value = *(Vector3D<Type>*)&i_lpBuffer[bufferPos];
			bufferPos += sizeof(this->value);

			this->SetValueX(value.x);
			this->SetValueY(value.y);
			this->SetValueZ(value.z);

			return bufferPos;
		}
		/** バッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
		S32 WriteToBuffer(BYTE* o_lpBuffer)const
		{
			U32 bufferPos = 0;

			// 値
			*(Vector3D<Type>*)&o_lpBuffer[bufferPos] = this->value;
			bufferPos += sizeof(this->value);

			return bufferPos;
		}

	public:
		//================================
		// 構造体を利用したデータの取り扱い.
		// 情報量が少ない代わりにアクセス速度が速い
		//================================

		/** 構造体に書き込む.
			@return	使用したバイト数. */
		S32 WriteToStruct(BYTE* o_lpBuffer)const
		{
			Vector3D<Type> value = this->GetValue();

			*(Vector3D<Type>*)o_lpBuffer = value;

			return sizeof(Vector3D<Type>);
		}
		/** 構造体から読み込む.
			@return	使用したバイト数. */
		S32 ReadFromStruct(const BYTE* i_lpBuffer)
		{
			Vector3D<Type> value = *(const Vector3D<Type>*)i_lpBuffer;

			this->SetValue(value);

			return sizeof(Vector3D<Type>);
		}
	};
	typedef Item_Vector3D<F32, ITEMTYPE_VECTOR3D_FLOAT> Item_Vector3D_Float;
	typedef Item_Vector3D<S32, ITEMTYPE_VECTOR3D_INT> Item_Vector3D_Int;

	/** 設定項目(Vector3)(実数)を作成する. */
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
	
	/** 設定項目(Vector3)(整数)を作成する. */
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