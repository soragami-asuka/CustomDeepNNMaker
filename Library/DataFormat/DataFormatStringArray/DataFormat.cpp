// DataFormatStringArray.cpp : DLL �A�v���P�[�V�����p�ɃG�N�X�|�[�g�����֐����`���܂��B
//

#include "stdafx.h"


#include"DataFormat.h"

#include<string>
#include<vector>
#include<set>
#include<map>


namespace Gravisbell {
namespace DataFormat {
namespace StringArray {
	
	struct BoolValue
	{
		F32 trueValue;	/**< true�̎��̒l */
		F32 falseValue;	/**< false�̎��̒l */

		BoolValue()
			:	trueValue	(1.0f)
			,	falseValue	(0.0f)
		{
		}
		BoolValue(F32 trueValue, F32 falseValue)
			:	trueValue	(trueValue)
			,	falseValue	(falseValue)
		{
		}
	};

	/** �f�[�^�t�H�[�}�b�g�̃A�C�e�� */
	class IDataFormatItem
	{
	public:
		/** �R���X�g���N�^ */
		IDataFormatItem(){}
		/** �f�X�g���N�^ */
		virtual ~IDataFormatItem(){}

	public:
		/** �g�p�o�b�t�@����Ԃ� */
		virtual U32 GetBufferCount()const = 0;

		/** �f�[�^��ǉ����� */
		virtual U32 AddValue(const std::wstring& value) = 0;

		/** ���K�� */
		virtual ErrorCode Normalize() = 0;
	};


	/** �f�[�^�t�H�[�}�b�g */
	class CDataFormat : public IDataFormat
	{
	private:
		std::wstring name;	/**< ���O */
		std::wstring text;	/**< ������ */

		std::set<std::wstring> lpCategoryName;	/**< �f�[�^��ʖ��ꗗ */

		std::map<std::wstring, BoolValue>	lpBoolValue;	/**< bool�l��F32�ɕϊ�����ݒ�l�̈ꗗ.	<�f�[�^��ʖ�, �ϊ��f�[�^> */

	public:
		/** �R���X�g���N�^ */
		CDataFormat()
		:	CDataFormat(L"", L"")
		{
		}
		/** �R���X�g���N�^ */
		CDataFormat(const wchar_t i_szName[], const wchar_t i_szText[])
		:	name	(i_szName)
		,	text	(i_szText)
		{
			// �f�t�H���g�l��ݒ肷��
			lpBoolValue[L""] = BoolValue();
		}
		/** �f�X�g���N�^ */
		virtual ~CDataFormat(){}

	public:
		/** ���O�̎擾 */
		const wchar_t* GetName()const
		{
			return name.c_str();
		}
		/** �������̎擾 */
		const wchar_t* GetText()const
		{
			return text.c_str();
		}

		/** X�����̗v�f�����擾 */
		U32 GetBufferCountX(const wchar_t i_szCategory[])const
		{
			return 1;
		}
		/** Y�����̗v�f�����擾 */
		U32 GetBufferCountY(const wchar_t i_szCategory[])const
		{
			return 1;
		}
		/** Z�����̗v�f�����擾 */
		U32 GetBufferCountZ(const wchar_t i_szCategory[])const
		{
			return 1;
		}
		/** CH�����̗v�f�����擾 */
		U32 GetBufferCountCH(const wchar_t i_szCategory[])const
		{
			return 0;
		}

		/** �f�[�^�\�����擾 */
		IODataStruct GetDataStruct(const wchar_t i_szCategory[])const
		{
			return IODataStruct(this->GetBufferCountCH(i_szCategory), this->GetBufferCountX(i_szCategory), this->GetBufferCountY(i_szCategory), this->GetBufferCountZ(i_szCategory));
		}

		/** �J�e�S���[�����擾���� */
		U32 GetCategoryCount()const
		{
			return this->lpCategoryName.size();
		}
		/** �J�e�S���[����ԍ��w��Ŏ擾���� */
		const wchar_t* GetCategoryNameByNum(U32 categoryNo)const
		{
			if(categoryNo >= this->lpCategoryName.size())
				return NULL;

			auto it = this->lpCategoryName.begin();
			for(U32 no=0; no<categoryNo; no++)
				it++;
			return it->c_str();
		}

	public:
		/** true�̏ꍇ�̒l�ݒ���擾���� */
		F32 GetTrueValue(const wchar_t i_szCategory[] = L"")const
		{
			// �J�e�S��������
			auto it = this->lpBoolValue.find(i_szCategory);
			if(it != this->lpBoolValue.end())
				return it->second.trueValue;

			// �f�t�H���g�l���ݒ肳��Ă��Ȃ��ꍇ��1.0��Ԃ�
			if((std::wstring)i_szCategory == L"")
				return 1.0f;

			// �f�t�H���g�l���Č���
			return GetTrueValue(L"");
		}
		/** false�̏ꍇ�̒l�ݒ���擾���� */
		F32 GetFalseValue(const wchar_t i_szCategory[] = L"")const
		{
			// �J�e�S��������
			auto it = this->lpBoolValue.find(i_szCategory);
			if(it != this->lpBoolValue.end())
				return it->second.falseValue;

			// �f�t�H���g�l���ݒ肳��Ă��Ȃ��ꍇ��0.0��Ԃ�
			if((std::wstring)i_szCategory == L"")
				return 0.0f;

			// �f�t�H���g�l���Č���
			return GetFalseValue(L"");
		}

	};

	/** ������̔z���ǂݍ��ރf�[�^�t�H�[�}�b�g���쐬���� */
	GRAVISBELL_LIBRAY_API IDataFormat* CreateDataFormat(const wchar_t i_szName[], const wchar_t i_szText[])
	{
		return new CDataFormat(i_szName, i_szText);
	}
	/** ������̔z���ǂݍ��ރf�[�^�t�H�[�}�b�g���쐬���� */
	extern GRAVISBELL_LIBRAY_API IDataFormat* CreateDataFormatFromXML(const wchar_t szXMLFilePath[])
	{
		CDataFormat* pDataFormat = new CDataFormat();

		return pDataFormat;
	}


}	// StringArray
}	// DataFormat
}	// Gravisbell


