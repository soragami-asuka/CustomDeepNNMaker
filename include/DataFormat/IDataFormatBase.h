//=======================================
// �f�[�^�t�H�[�}�b�g�̃x�[�X�C���^�[�t�F�[�X
//=======================================
#ifndef __GRAVISBELL_I_DATAFORMAT_BASE_H__
#define __GRAVISBELL_I_DATAFORMAT_BASE_H__

#include"../Common/Common.h"
#include"../Common/ErrorCode.h"
#include"../Common/IODataStruct.h"
#include"../Common/Guiddef.h"

#include"../SettingData/Standard/IData.h"

namespace Gravisbell {
namespace DataFormat {


	class IDataFormatBase
	{
	public:
		/** �R���X�g���N�^ */
		IDataFormatBase(){}
		/** �f�X�g���N�^ */
		virtual ~IDataFormatBase(){}

	public:
		/** ���O�̎擾 */
		virtual const wchar_t* GetName()const = 0;
		/** �������̎擾 */
		virtual const wchar_t* GetText()const = 0;

		/** X�����̗v�f�����擾 */
		virtual U32 GetBufferCountX(const wchar_t i_szCategory[])const = 0;
		/** Y�����̗v�f�����擾 */
		virtual U32 GetBufferCountY(const wchar_t i_szCategory[])const = 0;
		/** Z�����̗v�f�����擾 */
		virtual U32 GetBufferCountZ(const wchar_t i_szCategory[])const = 0;
		/** CH�����̗v�f�����擾 */
		virtual U32 GetBufferCountCH(const wchar_t i_szCategory[])const = 0;

		/** �f�[�^�\�����擾 */
		virtual IODataStruct GetDataStruct(const wchar_t i_szCategory[])const = 0;

		/** �J�e�S���[�����擾���� */
		virtual U32 GetCategoryCount()const = 0;
		/** �J�e�S���[����ԍ��w��Ŏ擾���� */
		virtual const wchar_t* GetCategoryNameByNum(U32 categoryNo)const = 0;



	public:
		/** �f�[�^���o�C�i���`���Œǉ�����.
			�o�C�i���`���̓t�H�[�}�b�g�̓��e�Ɋւ�炸[GetBufferCountZ()][GetBufferCountY()][GetBufferCountX()][GetBufferCountCH()]�̔z��f�[�^�̐擪�A�h���X��n��. */
//		virtual Gravisbell::ErrorCode AddDataByBinary(const F32 i_buffer[]) = 0;

		/** �f�[�^�����擾���� */
		virtual U32 GetDataCount()const = 0;

		/** �f�[�^���擾���� */
		virtual const F32* GetDataByNum(U32 i_dataNo, const wchar_t i_szCategory[])const = 0;

	public:
		/** ���K������.
			�f�[�^�̒ǉ����I��������A��x�̂ݎ��s. ��������s����ƒl�����������Ȃ�̂Œ���. */
		virtual Gravisbell::ErrorCode Normalize() = 0;
	};

}	// DataFormat
}	// Gravisbell



#endif	// __GRAVISBELL_I_DATAFORMAT_BASE_H__