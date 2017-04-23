//=======================================
// ��ʐݒ�
//=======================================
#ifndef __GRAVISBELL_COMMON_H__
#define __GRAVISBELL_COMMON_H__

#ifndef BYTE
typedef unsigned char BYTE;
#endif

namespace Gravisbell {



	typedef signed __int8  S08;
	typedef signed __int16 S16;
	typedef signed __int32 S32;
	typedef signed __int64 S64;

	typedef unsigned __int8  U08;
	typedef unsigned __int16 U16;
	typedef unsigned __int32 U32;
	typedef unsigned __int64 U64;

	typedef float	F32;
	typedef double	F64;

	template<class Type>
	struct Vector3D
	{
		Type x;
		Type y;
		Type z;

		Vector3D()
			:	x	(0)
			,	y	(0)
			,	z	(0)
		{
		}
		Vector3D(Type x, Type y, Type z)
			:	x	(x)
			,	y	(y)
			,	z	(z)
		{
		}

		bool operator==(const Vector3D& i_value)const
		{
			if(this->x != i_value.x)
				return false;
			if(this->y != i_value.y)
				return false;
			if(this->z != i_value.z)
				return false;
			return true;
		}
		bool operator!=(const Vector3D& i_value)const
		{
			return !(*this == i_value);
		}
			
	};

	template<class Type>
	struct Vector2D
	{
		Type x;
		Type y;

		Vector2D()
			:	x	(0)
			,	y	(0)
		{
		}
		Vector2D(Type x, Type y)
			:	x	(x)
			,	y	(y)
		{
		}
	};

	/** ���C���[�Ԃ̃f�[�^�̂������s���o�b�`�����p2�����z��|�C���^�^.
		[�o�b�`�T�C�Y][�o�b�t�@��] */
	typedef F32**				BATCH_BUFFER_POINTER;
	/** ���C���[�Ԃ̃f�[�^�̂������s���o�b�`�����p2�����z��|�C���^�^(�萔).
		[�o�b�`�T�C�Y][�o�b�t�@��] */
	typedef const F32*const*	CONST_BATCH_BUFFER_POINTER;

}	// Gravisbell


#endif // __GRAVISBELL_COMMON_H__